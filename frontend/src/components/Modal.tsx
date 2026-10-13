import {useEffect, useRef} from 'react'
import type {ReactNode} from 'react'
import {motion} from 'motion/react'
import {X} from 'lucide-react'
export default function Modal({title, onClose, children}: {title: string; onClose: () => void; children: ReactNode}) {
  const dialog = useRef<HTMLDivElement>(null)
  const close = useRef(onClose); close.current = onClose
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null
    const node = dialog.current!
    node.querySelector<HTMLButtonElement>('button')?.focus()
    function keydown(event: KeyboardEvent) {
      if (event.key === 'Escape') close.current()
      if (event.key !== 'Tab') return
      const items = node.querySelectorAll<HTMLElement>('button:not([disabled]), [href], input:not([disabled]), select, [tabindex="0"]')
      const first = items[0], last = items[items.length - 1]
      if (event.shiftKey && document.activeElement === first) {event.preventDefault(); last.focus()}
      else if (!event.shiftKey && document.activeElement === last) {event.preventDefault(); first.focus()}
    }
    document.addEventListener('keydown', keydown)
    const previousOverflow = document.body.style.overflow; document.body.style.overflow = 'hidden'
    return () => {document.removeEventListener('keydown', keydown); document.body.style.overflow = previousOverflow; previous?.focus()}
  }, [])
  return <motion.div className="modal-backdrop" initial={{opacity: 0}} animate={{opacity: 1}} exit={{opacity: 0}} onClick={onClose}>
    <motion.div ref={dialog} className="modal" role="dialog" aria-modal="true" aria-label={title} initial={{opacity: 0, y: 12, scale: 0.98}} animate={{opacity: 1, y: 0, scale: 1}} exit={{opacity: 0, y: 8}} onClick={event => event.stopPropagation()}>
      <div className="modal-heading"><h2>{title}</h2><button className="icon-button" onClick={onClose} aria-label="Close dialog"><X size={20}/></button></div>{children}
    </motion.div>
  </motion.div>
}
