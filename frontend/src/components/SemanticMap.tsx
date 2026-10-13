import {Component, lazy, Suspense, useEffect, useState} from 'react'
import type {ReactNode} from 'react'
import type {Game, MovePreview} from '../lib/types'
import SemanticMap2D from './SemanticMap2D'
import '../styles/semantic-map.css'

const SemanticMap3D = lazy(() => import('./SemanticMap3D'))
export type ViewMode = '2d' | '3d'
export interface SemanticMapProps {
  game: Game; labels: boolean; reduced: boolean; preview?: MovePreview | null
  viewMode?: ViewMode; onViewModeChange?: (mode: ViewMode) => void
  followPlayer?: boolean; replayKey?: number
}

function supportsWebGL(): boolean {
  try {
    const canvas = document.createElement('canvas')
    const context = canvas.getContext('webgl2')
    if (!context) return false
    context.getExtension('WEBGL_lose_context')?.loseContext()
    return true
  } catch {return false}
}

class MapBoundary extends Component<{children: ReactNode; fallback: ReactNode; onFailure: () => void}, {failed: boolean}> {
  state = {failed: false}
  static getDerivedStateFromError() {return {failed: true}}
  componentDidCatch() {this.props.onFailure()}
  render() {return this.state.failed ? this.props.fallback : this.props.children}
}

export default function SemanticMap(props: SemanticMapProps) {
  const [mode, setMode] = useState<ViewMode>(() => window.matchMedia('(min-width: 768px)').matches ? '3d' : '2d')
  const [webGL, setWebGL] = useState<boolean | null>(null)
  const requestedMode = props.viewMode ?? mode
  // Only probe when a 3D view is requested; the 2D/mobile path needs no GPU.
  useEffect(() => {if (requestedMode === '3d' && webGL === null) setWebGL(supportsWebGL())}, [requestedMode, webGL])
  const show3D = requestedMode === '3d' && webGL === true && !!props.game.map3d
  const fallback = requestedMode === '3d' && (webGL === false || !props.game.map3d)
  const changeMode = (next: ViewMode) => {setMode(next); props.onViewModeChange?.(next)}
  const map2D = <SemanticMap2D {...props}/>
  return <div className="semantic-map-shell" data-view-mode={show3D ? '3d' : '2d'}>
    {show3D ? <MapBoundary onFailure={() => setWebGL(false)} fallback={map2D}><Suspense fallback={<div className="semantic-map map-loading">Opening semantic space…</div>}><SemanticMap3D {...props} onUnavailable={() => setWebGL(false)}/></Suspense></MapBoundary> : map2D}
    <div className="map-view-toggle" role="group" aria-label="Map dimensions">
      <button type="button" aria-pressed={!show3D} onClick={() => changeMode('2d')}>2D</button>
      <button type="button" aria-pressed={show3D} onClick={() => changeMode('3d')} disabled={webGL === false || !props.game.map3d} title={webGL === false ? '3D requires WebGL; the 2D map is available.' : 'Explore in three dimensions'}>3D</button>
    </div>
    {fallback && <p className="map-fallback" role="status">{webGL === false ? 'WebGL is unavailable. Showing the 2D map.' : 'Showing the 2D map for this saved round.'}</p>}
  </div>
}
