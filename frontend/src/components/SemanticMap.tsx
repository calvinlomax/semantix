import {useEffect, useMemo, useRef, useState} from 'react'
import {select, zoom, zoomIdentity, scaleLinear, extent, line} from 'd3'
import type {ZoomBehavior, ZoomTransform} from 'd3'
import {motion, useReducedMotion} from 'motion/react'
import {Expand, Minus, Plus, LocateFixed} from 'lucide-react'
import type {Game, Point} from '../lib/types'

export default function SemanticMap({game, labels, reduced}: {game: Game; labels: boolean; reduced: boolean}) {
  const svg = useRef<SVGSVGElement>(null)
  const zoomer = useRef<ZoomBehavior<SVGSVGElement, unknown> | null>(null)
  const [transform, setTransform] = useState<ZoomTransform>(zoomIdentity)
  const prefersReduced = useReducedMotion()
  const animate = !(reduced || prefersReduced)
  // A round's initial map anchors define the coordinate frame for its entire lifetime.
  const anchors = useMemo(() => [game.map.start, game.map.target, ...game.map.landmarks.map(l => l.position)], [game.id]) // eslint-disable-line react-hooks/exhaustive-deps
  const {x, y} = useMemo(() => {
    const ex = extent(anchors, p => p[0]) as [number, number], ey = extent(anchors, p => p[1]) as [number, number]
    const span = Math.max(ex[1] - ex[0], (ey[1] - ey[0]) * 2.15, 0.1)
    const cx = (ex[0] + ex[1]) / 2, cy = (ey[0] + ey[1]) / 2
    return {x: scaleLinear().domain([cx - span / 2, cx + span / 2]).range([112, 888]), y: scaleLinear().domain([cy - span / 4.3, cy + span / 4.3]).range([344, 82])}
  }, [anchors])
  const point = (p: Point) => ({x: x(p[0]), y: y(p[1])})
  const start = point(game.map.start), target = point(game.map.target), player = point(game.map.player)
  const submittedWords = [...new Map(game.history.map(move => [move.word, move])).values()]
  const occupied = [
    {left: start.x - game.start_word.length * 5, right: start.x + game.start_word.length * 5, top: start.y - 43, bottom: start.y + 12},
    {left: target.x - Math.max(55, game.target_word.length * 5), right: target.x + Math.max(55, game.target_word.length * 5), top: target.y - 66, bottom: target.y + 22},
  ]
  const labelPositions = new Map<string, {x: number; y: number}>()
  const labelWords = [...game.map.landmarks, ...submittedWords.map(move => ({word: move.word, position: move.word_position}))]
  for (const landmark of labelWords) {
    if (labelPositions.has(landmark.word)) continue
    const p = point(landmark.position), width = landmark.word.length * 7 + 4
    for (const [dx, dy] of [[8, 4], [8, -12], [-width - 8, 4], [8, 20], [-width - 8, -12]]) {
      const box = {left: p.x + dx, right: p.x + dx + width, top: p.y + dy - 12, bottom: p.y + dy + 4}
      if (box.left < 25 || box.right > 975 || box.top < 55 || box.bottom > 380) continue
      if (occupied.some(b => box.left < b.right + 5 && box.right > b.left - 5 && box.top < b.bottom + 3 && box.bottom > b.top - 3)) continue
      occupied.push(box); labelPositions.set(landmark.word, {x: box.left, y: p.y + dy}); break
    }
  }
  const path = line<Point>().x(p => x(p[0])).y(p => y(p[1]))(game.map.trail) ?? ''
  useEffect(() => {
    if (!svg.current) return
    const behavior = zoom<SVGSVGElement, unknown>().scaleExtent([0.65, 4]).filter(event => !event.ctrlKey && (event.type !== 'wheel' || event.shiftKey)).on('zoom', event => setTransform(event.transform))
    zoomer.current = behavior
    select(svg.current).call(behavior).call(behavior.transform, zoomIdentity).on('dblclick.zoom', null)
    setTransform(zoomIdentity)
    return () => {if (svg.current) select(svg.current).on('.zoom', null)}
  }, [game.id])
  function zoomBy(value: number) {if (svg.current && zoomer.current) select(svg.current).call(zoomer.current.scaleBy, value)}
  function reset() {if (svg.current && zoomer.current) select(svg.current).call(zoomer.current.transform, zoomIdentity)}
  return <div className="semantic-map">
    <div className="map-topline"><span><span className="live-dot"/> SEMANTIC SPACE</span><span className="map-dimension">300D → 2D</span></div>
    <svg ref={svg} viewBox="0 0 1000 420" className="map-svg" role="img" aria-label={`Semantic map from ${game.start_word} to ${game.target_word}. Your similarity is ${game.similarity.toFixed(3)}. Drag to pan; use the zoom buttons to explore.`}>
      <defs>
        <pattern id="map-grid" width="32" height="32" patternUnits="userSpaceOnUse"><circle cx="1" cy="1" r="0.75" fill="#494252" opacity="0.5"/></pattern>
        <radialGradient id="target-glow"><stop offset="0%" stopColor="#a38bf8" stopOpacity="0.17"/><stop offset="100%" stopColor="#a38bf8" stopOpacity="0"/></radialGradient>
        <radialGradient id="start-glow"><stop offset="0%" stopColor="#77dec8" stopOpacity="0.12"/><stop offset="100%" stopColor="#77dec8" stopOpacity="0"/></radialGradient>
        <linearGradient id="trail-color"><stop offset="0%" stopColor="#7bdec8"/><stop offset="100%" stopColor="#b39afa"/></linearGradient>
        <filter id="point-glow"><feGaussianBlur stdDeviation="3"/></filter>
      </defs>
      <rect width="1000" height="420" fill="url(#map-grid)"/>
      <g transform={transform.toString()}>
        <circle cx={target.x} cy={target.y} r="140" fill="url(#target-glow)"/>
        <circle cx={start.x} cy={start.y} r="100" fill="url(#start-glow)"/>
        <line x1={start.x} y1={start.y} x2={target.x} y2={target.y} stroke="#9586b1" strokeDasharray="3 8" opacity="0.14"/>
        {game.map.landmarks.filter(l => ![game.start_word, game.target_word].includes(l.word)).map((landmark, i) => {
          const p = point(landmark.position)
          const label = labelPositions.get(landmark.word)
          return <g key={landmark.word} className="landmark"><title>{landmark.word}</title><circle cx={p.x} cy={p.y} r={i % 3 === 0 ? 2.7 : 2} fill={i % 3 === 0 ? '#a899c9' : '#6c657e'}/>{labels && label && !submittedWords.some(move => move.word === landmark.word) && <text x={label.x} y={label.y} fontSize="12" fill="#a49bae">{landmark.word.replaceAll('_', ' ')}</text>}</g>
        })}
        {submittedWords.map(move => {const p = point(move.word_position), label = labelPositions.get(move.word); return <g key={`word-${move.word}`}><title>{move.word}</title><circle cx={p.x} cy={p.y} r="3" fill="#d5c6ff" opacity="0.65"/>{labels && label && <text x={label.x} y={label.y} fontSize="12" fill="#d3c3ee">{move.word.replaceAll('_', ' ')}</text>}</g>})}
        <motion.path d={path} stroke="url(#trail-color)" strokeWidth="2.5" fill="none" strokeLinecap="round" initial={false} animate={{pathLength: 1}} transition={{duration: animate ? 0.7 : 0}}/>
        {game.map.trail.slice(1, -1).map((p, i) => <circle key={i} cx={x(p[0])} cy={y(p[1])} r="4" fill="#181620" stroke="#b29acb" strokeWidth="1.5"/>)}
        <g><circle cx={start.x} cy={start.y} r="12" fill="#7bdec8" opacity="0.08"/><circle cx={start.x} cy={start.y} r="5" fill="#7bdec8"/>
          <text x={start.x} y={start.y - 29} textAnchor="middle" fontSize="10" fill="#71bbaa" letterSpacing="1.7">{game.moves_used ? 'START' : 'START · YOU'}</text>
          <text x={start.x} y={start.y - 12} textAnchor="middle" fontSize="15" fill="#b1e5da" fontWeight="500">{game.start_word.replaceAll('_', ' ')}</text>
        </g>
        {game.status === 'won' && animate && Array.from({length: 12}, (_, index) => {
          const angle = index * Math.PI / 6
          return <motion.circle key={`spark-${game.id}-${index}`} cx={target.x} cy={target.y} r="2.5" fill={index % 2 ? '#bda2fa' : '#8de4d5'} initial={{x: 0, y: 0, opacity: 0}} animate={{x: Math.cos(angle) * 90, y: Math.sin(angle) * 90, opacity: [0, .8, 0]}} transition={{duration: 1.5, ease: 'easeOut', delay: index * .025}}/>
        })}
        <g><motion.circle cx={target.x} cy={target.y} r="30" fill="none" stroke="#b19bea" strokeWidth="1" animate={animate ? {r: [27, 34, 27], opacity: [0.15, 0.3, 0.15]} : {opacity: 0.2}} transition={{duration: 4, repeat: Infinity}}/>
          <circle cx={target.x} cy={target.y} r="18" fill="#ad8deb" fillOpacity="0.06" stroke="#a98bdc" strokeOpacity="0.3" strokeDasharray="3 4"/>
          <circle cx={target.x} cy={target.y} r="7" fill="#ba9bf7"/><circle cx={target.x} cy={target.y} r="3" fill="#e9ddff"/>
          <text x={target.x} y={target.y - 49} textAnchor="middle" fontSize="10" fill="#a99ac8" letterSpacing="1.7">DESTINATION</text>
          <text x={target.x} y={target.y - 30} textAnchor="middle" fontSize="17" fill="#e3d6ff" fontWeight="500">{game.target_word.replaceAll('_', ' ')}</text>
        </g>
        {game.moves_used > 0 && <motion.g initial={false} animate={{x: player.x, y: player.y}} transition={{type: 'spring', stiffness: 90, damping: 22, duration: animate ? undefined : 0}}>
          <circle r="15" fill="#d9c7ff" opacity="0.09"/><circle r="8" fill="#b99afa" filter="url(#point-glow)" opacity="0.7"/><circle r="5.5" fill="#f3eaff" stroke="#ac8ee4" strokeWidth="1.5"/>
          <text x="0" y="25" textAnchor="middle" fontSize="9" fill="#ded2f5" letterSpacing="1.5">YOU</text>
        </motion.g>}
      </g>
    </svg>
    <div className="map-bottomline"><div className="map-key"><span><i className="key-dot key-start"/> Start</span><span><i className="key-dot key-you"/> You</span><span><i className="key-dot key-target"/> Target</span></div><div className="map-controls"><button aria-label="Zoom out" onClick={() => zoomBy(0.8)}><Minus size={15}/></button><button aria-label="Zoom in" onClick={() => zoomBy(1.25)}><Plus size={15}/></button><button aria-label="Reset map view" onClick={reset}><Expand size={15}/></button></div></div>
    <div className="map-pan-hint"><LocateFixed size={12}/> Drag to explore · Shift + scroll to zoom</div>
  </div>
}
