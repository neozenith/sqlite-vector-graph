/** FCoSE layout controls: sliders for node repulsion, edge length, gravity + re-run button. */

import { Button } from '@/components/ui/button'
import type { FcoseParams } from '@/lib/fcose'

interface Props {
  params: FcoseParams
  onChange: (params: FcoseParams) => void
  onRunLayout: () => void
}

export function GraphControls({ params, onChange, onRunLayout }: Props) {
  return (
    <div className="space-y-2 text-xs">
      <label className="flex items-center justify-between">
        <span className="text-muted-foreground">Node repulsion</span>
        <span className="font-mono w-12 text-right">{params.nodeRepulsion}</span>
      </label>
      <input
        type="range"
        min={500}
        max={20000}
        step={500}
        value={params.nodeRepulsion}
        onChange={(e) => onChange({ ...params, nodeRepulsion: Number(e.target.value) })}
        className="w-full h-1.5 accent-primary"
      />

      <label className="flex items-center justify-between">
        <span className="text-muted-foreground">Edge length</span>
        <span className="font-mono w-12 text-right">{params.idealEdgeLength}</span>
      </label>
      <input
        type="range"
        min={10}
        max={200}
        step={5}
        value={params.idealEdgeLength}
        onChange={(e) => onChange({ ...params, idealEdgeLength: Number(e.target.value) })}
        className="w-full h-1.5 accent-primary"
      />

      <label className="flex items-center justify-between">
        <span className="text-muted-foreground">Gravity</span>
        <span className="font-mono w-12 text-right">{params.gravity.toFixed(2)}</span>
      </label>
      <input
        type="range"
        min={0}
        max={2}
        step={0.05}
        value={params.gravity}
        onChange={(e) => onChange({ ...params, gravity: Number(e.target.value) })}
        className="w-full h-1.5 accent-primary"
      />

      <Button variant="outline" size="sm" className="w-full h-7 text-xs mt-1" onClick={onRunLayout}>
        Re-run Layout
      </Button>
    </div>
  )
}
