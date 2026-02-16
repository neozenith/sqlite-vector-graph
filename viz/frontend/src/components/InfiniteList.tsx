/** Infinite-scroll list using IntersectionObserver sentinel. No text truncation. */

import { useEffect, useRef } from 'react'
import { Input } from '@/components/ui/input'

interface Props {
  items: Record<string, unknown>[]
  total: number
  hasNextPage: boolean
  isFetchingNextPage: boolean
  isLoading: boolean
  filter: string
  onFilterChange: (query: string) => void
  onLoadMore: () => void
}

export function InfiniteList({
  items,
  total,
  hasNextPage,
  isFetchingNextPage,
  isLoading,
  filter,
  onFilterChange,
  onLoadMore,
}: Props) {
  const sentinelRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = sentinelRef.current
    if (!el) return
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && hasNextPage && !isFetchingNextPage) {
          onLoadMore()
        }
      },
      { threshold: 0.1 },
    )
    observer.observe(el)
    return () => observer.disconnect()
  }, [hasNextPage, isFetchingNextPage, onLoadMore])

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <Input
          placeholder="Filter..."
          value={filter}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) => onFilterChange(e.target.value)}
          className="h-8 text-xs"
        />
        <span className="text-xs text-muted-foreground whitespace-nowrap">{total.toLocaleString()} total</span>
      </div>

      {isLoading ? (
        <div className="text-xs text-muted-foreground py-4 text-center">Loading...</div>
      ) : items.length === 0 ? (
        <div className="text-xs text-muted-foreground py-4 text-center">No items found</div>
      ) : (
        <div className="space-y-1">
          {items.map((item, i) => (
            <div key={i} className="text-xs p-2 bg-muted rounded">
              {Object.entries(item).map(([key, value]) => (
                <span key={key} className="mr-3 inline-block">
                  <span className="text-muted-foreground">{key}:</span>{' '}
                  <span className="font-mono break-words whitespace-pre-wrap">{String(value)}</span>
                </span>
              ))}
            </div>
          ))}
          <div ref={sentinelRef} className="h-4" />
          {isFetchingNextPage && <div className="text-xs text-muted-foreground text-center py-2">Loading more...</div>}
        </div>
      )}
    </div>
  )
}
