import {useEffect, useRef, useState} from 'react'

/** Animate only display coordinates. Every endpoint comes from the server. */
export function useTrajectory<T extends number[]>(trail: T[], reduced: boolean, replayKey = 0): T[] {
  const [visible, setVisible] = useState(trail)
  const lastReplay = useRef(replayKey)
  useEffect(() => {
    const replay = replayKey !== lastReplay.current
    lastReplay.current = replayKey
    if (reduced || trail.length < 2) {setVisible(trail); return}
    const firstSegment = replay ? 0 : trail.length - 2
    const duration = replay ? Math.min(2200, (trail.length - 1) * 450) : 650
    let frame = 0
    const start = performance.now()
    const tick = (now: number) => {
      // A frame timestamp can precede an effect's performance.now() within
      // the same frame. Clamp both ends before indexing the trajectory.
      const progress = Math.max(0, Math.min(1, (now - start) / duration))
      if (progress >= 1) {setVisible(trail); return}
      const segment = firstSegment + progress * (trail.length - 1 - firstSegment)
      const index = Math.floor(segment), amount = segment - index
      const point = trail[index].map((value, axis) => value + (trail[index + 1][axis] - value) * amount) as T
      setVisible([...trail.slice(0, index + 1), point])
      frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [trail, reduced, replayKey])
  return visible
}
