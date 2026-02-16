import { NavLink } from 'react-router-dom'
import { STAGE_NAMES, KG_STAGE_ROUTES } from '@/lib/constants'

const PILL_ITEMS = [
  { to: '/kg', label: 'Overview', end: true },
  ...Object.entries(STAGE_NAMES).map(([num, name]) => ({
    to: `/kg/${KG_STAGE_ROUTES[Number(num)]}/`,
    label: name,
    end: false,
  })),
  { to: '/kg/query/', label: 'Query', end: false },
]

export function KGStagePills() {
  return (
    <div className="flex items-center gap-1 px-4 py-2 border-b overflow-x-auto shrink-0">
      {PILL_ITEMS.map((item) => (
        <NavLink
          key={item.to}
          to={item.to}
          end={item.end}
          className={({ isActive }) =>
            `whitespace-nowrap rounded-full px-3 py-1 text-xs font-medium transition-colors ${
              isActive ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground hover:bg-muted/80'
            }`
          }
        >
          {item.label}
        </NavLink>
      ))}
    </div>
  )
}
