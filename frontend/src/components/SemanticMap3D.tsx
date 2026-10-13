import {useEffect, useMemo, useRef, useState} from 'react'
import type {KeyboardEvent, RefObject} from 'react'
import {Canvas, useFrame, useThree} from '@react-three/fiber'
import {Html, Line, OrbitControls} from '@react-three/drei'
import type {OrbitControls as OrbitControlsImpl} from 'three-stdlib'
import {Vector3} from 'three'
import {Expand, LocateFixed, Minus, Plus} from 'lucide-react'
import {useReducedMotion} from 'motion/react'
import type {Point3D} from '../lib/types'
import type {SemanticMapProps} from './SemanticMap'
import {useTrajectory} from './useTrajectory'

const colors = {start: '#80ddc3', target: '#be92ff', player: '#f5e8ff', trail: '#bda0f0', landmark: '#847094'}
type ControlsRef = RefObject<OrbitControlsImpl | null>

function PointMarker({position, color, radius = .045, word, kind = '', caption, captionRef, similarity, move, glow = false}: {
  position: Point3D; color: string; radius?: number; word: string; kind?: string
  caption?: string; captionRef?: RefObject<HTMLDivElement | null>; similarity?: number; move?: number; glow?: boolean
}) {
  const [hovered, setHovered] = useState(false)
  return <group position={position}>
    {glow && <mesh><sphereGeometry args={[radius * 2.8, 16, 12]}/><meshBasicMaterial color={color} transparent opacity={.07} depthWrite={false}/></mesh>}
    <mesh onPointerOver={event => {event.stopPropagation(); setHovered(true)}} onPointerOut={() => setHovered(false)}>
      <sphereGeometry args={[radius, 16, 12]}/><meshBasicMaterial color={color}/>
    </mesh>
    {caption && <Html center style={{pointerEvents: 'none'}} zIndexRange={[1, 0]}><div ref={captionRef} className={`map-label ${kind}`}>{kind === 'start' && <em>START</em>}{kind === 'target' && <em>DESTINATION</em>}{caption.replaceAll('_', ' ')}</div></Html>}
    {hovered && <Html center style={{pointerEvents: 'none'}} zIndexRange={[5, 4]}><div className="map-tooltip">{word.replaceAll('_', ' ')}{move !== undefined && <small>Move {move}</small>}{similarity !== undefined && <small>Similarity {similarity.toFixed(3)}</small>}</div></Html>}
  </group>
}

function LandmarkMarkers({points, reserved, labels}: {
  points: {word: string; position: Point3D; similarity?: number}[]
  reserved: {word: string; position: Point3D; above: number; below: number}[]; labels: boolean
}) {
  const {camera, size} = useThree()
  const captions = useMemo(() => points.map(() => ({current: null as HTMLDivElement | null})), [points.length])
  const vector = useMemo(() => new Vector3(), [])
  useFrame(() => {
    if (!labels) return
    // At most twelve labels: only screen-space boxes, never embedding/PCA work.
    camera.updateMatrixWorld()
    const screen = (position: Point3D) => {
      vector.set(...position).project(camera)
      return {x: (vector.x + 1) * size.width / 2, y: (1 - vector.y) * size.height / 2, z: vector.z}
    }
    const occupied = reserved.map(point => {
      const {x, y} = screen(point.position), width = Math.max(60, point.word.length * 8 + 18)
      return {left: x - width / 2, right: x + width / 2, top: y - point.above, bottom: y + point.below}
    })
    let mask = 0
    points.forEach((point, index) => {
      const {x, y, z} = screen(point.position), width = point.word.length * 6 + 16
      const box = {left: x - width / 2, right: x + width / 2, top: y - 34, bottom: y - 7}
      if (z < -1 || z > 1 || box.left < 12 || box.right > size.width - 12 || box.top < 32 || box.bottom > size.height - 45) return
      if (occupied.some(other => box.left < other.right && box.right > other.left && box.top < other.bottom && box.bottom > other.top)) return
      occupied.push(box)
      mask |= 1 << index
    })
    // Keep HTML portals mounted; camera movement only changes visibility.
    captions.forEach((caption, index) => {
      if (caption.current) caption.current.style.visibility = mask & (1 << index) ? 'visible' : 'hidden'
    })
  })
  return points.map((point, index) => <PointMarker key={point.word} position={point.position} color={colors.landmark} radius={.023} word={point.word} caption={labels ? point.word : undefined} captionRef={captions[index]} similarity={point.similarity}/>)
}

function CameraTracking({controls, player, follow, resetKey, onUnavailable}: {controls: ControlsRef; player: Point3D; follow: boolean; resetKey: string; onUnavailable: () => void}) {
  const {camera, gl, invalidate, size} = useThree()
  const previous = useRef<Point3D>(player)
  const wasFollowing = useRef(false)
  const initialPosition = useMemo(() => new Vector3(0, 1, size.width < size.height * 1.2 ? 15 : 10), [size.width, size.height])
  useEffect(() => {
    camera.position.copy(initialPosition)
    controls.current?.target.set(0, 0, 0)
    controls.current?.update()
    controls.current?.saveState()
    previous.current = player
    invalidate()
  }, [resetKey, camera, controls, invalidate, initialPosition]) // Player changes must not reset a user's camera.
  useEffect(() => {
    if (follow && controls.current) {
      const next = new Vector3(...player)
      const delta = next.clone().sub(wasFollowing.current ? new Vector3(...previous.current) : controls.current.target)
      controls.current.target.copy(next)
      camera.position.add(delta)
      controls.current.update()
      invalidate()
    }
    wasFollowing.current = follow
    previous.current = player
  }, [player, follow, controls, camera, invalidate])
  useEffect(() => {
    const lost = (event: Event) => {event.preventDefault(); onUnavailable()}
    gl.domElement.addEventListener('webglcontextlost', lost)
    return () => gl.domElement.removeEventListener('webglcontextlost', lost)
  }, [gl, onUnavailable])
  return null
}

function Scene({game, labels, reduced, preview, followPlayer, replayKey, controls, resetKey, onUnavailable}: SemanticMapProps & {controls: ControlsRef; resetKey: string; onUnavailable: () => void}) {
  const map = game.map3d!
  const prefersReduced = useReducedMotion()
  const still = reduced || !!prefersReduced
  // Round anchors fix both the PCA basis and its display scale. Camera changes
  // cannot trigger projection fitting, similarity work, or movement arithmetic.
  const project = useMemo(() => {
    const anchors = [map.start, map.target, ...map.landmarks.map(point => point.position)]
    const minimum = [0, 1, 2].map(axis => Math.min(...anchors.map(point => point[axis])))
    const maximum = [0, 1, 2].map(axis => Math.max(...anchors.map(point => point[axis])))
    const center = minimum.map((value, axis) => (value + maximum[axis]) / 2)
    const scale = 5 / Math.max(...maximum.map((value, axis) => value - minimum[axis]), .1)
    return (point: Point3D): Point3D => point.map((value, axis) => (value - center[axis]) * scale) as Point3D
  }, [game.id])
  const trajectory = useTrajectory(map.trail, still, replayKey)
  const trail = useMemo(() => trajectory.map(project), [trajectory, project])
  const player = trail.at(-1) ?? project(map.player)
  const start = project(map.start), target = project(map.target)
  const submitted = [...new Map(game.history.filter(move => move.word_position3d).map(move => [move.word, move])).values()]
  const landmarks = map.landmarks.filter(point => ![game.start_word, game.target_word, ...submitted.map(move => move.resolved_word ?? move.word)].includes(point.word)).slice(0, 12)
  return <>
    <color attach="background" args={['#14121b']}/>
    <gridHelper args={[14, 20, '#332a42', '#241f30']} position={[0, -3, 0]}/>
    <Line points={[start, target]} color="#625473" lineWidth={.7} transparent opacity={.35} dashed dashSize={.025} gapSize={.09}/>
    <LandmarkMarkers labels={labels} points={landmarks.map(point => ({...point, position: project(point.position)}))} reserved={[{word: game.start_word, position: start, above: 52, below: 10}, {word: game.target_word, position: target, above: 65, below: 10}, {word: 'YOU', position: player, above: 0, below: 38}, ...submitted.map(move => ({word: move.word, position: project(move.word_position3d!), above: 35, below: 10}))]}/>
    {submitted.map(move => <PointMarker key={move.word} position={project(move.word_position3d!)} color="#bea4df" radius={.031} word={move.word} caption={move.word} kind="submitted" move={move.number} similarity={move.word_similarity}/>)}
    {trail.length > 1 && <Line points={trail} color={colors.trail} lineWidth={2.3}/>} 
    {trail.slice(1, -1).map((position, index) => <PointMarker key={index} position={position} radius={.034} color="#a183bd" word={game.history[index]?.word ?? 'Position'} move={index + 1} similarity={game.history[index]?.similarity_after}/>)}
    <PointMarker position={start} color={colors.start} radius={.063} word={game.start_word} caption={game.start_word} kind="start" similarity={game.initial_similarity} glow/>
    <PointMarker position={target} color={colors.target} radius={.088} word={game.target_word} caption={game.target_word} kind="target" similarity={1} glow/>
    <group position={target}><mesh><sphereGeometry args={[.28, 20, 16]}/><meshBasicMaterial color={colors.target} wireframe transparent opacity={.075} depthWrite={false}/></mesh></group>
    <PointMarker position={player} color={colors.player} radius={.065} word="Current position" caption="YOU" kind="player" similarity={game.similarity} glow/>
    {preview && <><Line points={[project(map.player), project(preview.position3d)]} color="#d1baff" dashed dashSize={.07} gapSize={.055} lineWidth={1.4} transparent opacity={.65}/><group position={project(preview.position3d)}><mesh><sphereGeometry args={[.085, 16, 12]}/><meshBasicMaterial color="#d9c2ff" transparent opacity={.27} wireframe/></mesh><Html center style={{pointerEvents: 'none'}} zIndexRange={[1, 0]}><div className="map-label preview">{preview.delta >= 0 ? '+' : ''}{preview.delta.toFixed(3)}</div></Html></group></>}
    <OrbitControls ref={controls} makeDefault enableDamping={!still} dampingFactor={.12} minDistance={2} maxDistance={30} enablePan screenSpacePanning/>
    <CameraTracking controls={controls} player={player} follow={!!followPlayer} resetKey={resetKey} onUnavailable={onUnavailable}/>
  </>
}

export default function SemanticMap3D(props: SemanticMapProps & {onUnavailable: () => void}) {
  const controls = useRef<OrbitControlsImpl>(null)
  const [resetVersion, setResetVersion] = useState(0)
  const zoomBy = (factor: number) => {
    const orbit = controls.current
    if (!orbit) return
    orbit.object.position.sub(orbit.target).multiplyScalar(factor).add(orbit.target)
    orbit.update()
  }
  const reset = () => setResetVersion(value => value + 1)
  const keyboard = (event: KeyboardEvent<HTMLDivElement>) => {
    const orbit = controls.current
    if (!orbit) return
    if (event.key === '+' || event.key === '=') {event.preventDefault(); zoomBy(.85)}
    else if (event.key === '-') {event.preventDefault(); zoomBy(1.15)}
    else if (event.key.toLowerCase() === 'r' || event.key === 'Home') {event.preventDefault(); reset()}
    else if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) {
      event.preventDefault()
      const offset = orbit.object.position.clone().sub(orbit.target)
      const horizontal = event.key === 'ArrowLeft' || event.key === 'ArrowRight'
      const axis = horizontal ? new Vector3(0, 1, 0) : new Vector3().crossVectors(offset, orbit.object.up).normalize()
      offset.applyAxisAngle(axis, ['ArrowLeft', 'ArrowUp'].includes(event.key) ? .14 : -.14)
      orbit.object.position.copy(orbit.target).add(offset)
      orbit.update()
    }
  }
  return <div className="semantic-map map-3d">
    <div className="map-topline"><span><span className="live-dot"/> SEMANTIC SPACE</span><span className="map-dimension">300D → 3D</span></div>
    <div className="map-canvas" role="img" tabIndex={0} onKeyDown={keyboard} aria-label={`3D semantic map from ${props.game.start_word} to ${props.game.target_word}. Similarity ${props.game.similarity.toFixed(3)}. Drag to rotate, scroll to zoom, right drag to pan. Arrow keys rotate, plus and minus zoom, R resets.`}>
      <Canvas camera={{position: [0, 1, 10], fov: 43, near: .1, far: 80}} dpr={[1, 1.5]} frameloop="demand" gl={{antialias: true, alpha: false, powerPreference: 'low-power'}} fallback={<div className="map-loading">3D is unavailable. Choose 2D above.</div>}>
        <Scene {...props} controls={controls} resetKey={`${props.game.id}-${resetVersion}`}/>
      </Canvas>
    </div>
    <div className="map-bottomline"><div className="map-key"><span><i className="key-dot key-start"/> Start</span><span><i className="key-dot key-you"/> You</span><span><i className="key-dot key-target"/> Target</span></div><div className="map-controls"><button type="button" aria-label="Zoom out" onClick={() => zoomBy(1.2)}><Minus size={15}/></button><button type="button" aria-label="Zoom in" onClick={() => zoomBy(.8)}><Plus size={15}/></button><button type="button" aria-label="Reset map view" onClick={reset}><Expand size={15}/></button></div></div>
    <div className="map-pan-hint"><LocateFixed size={12}/> Drag to rotate · Scroll to zoom · Right drag to pan</div>
  </div>
}
