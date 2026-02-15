/** Generic paginated list: search input, prev/next controls, renders items as rows. */

import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';

interface Props {
  items: Record<string, unknown>[];
  total: number;
  page: number;
  pageSize: number;
  isLoading: boolean;
  filter: string;
  onFilterChange: (query: string) => void;
  onPageChange: (page: number) => void;
}

export function PaginatedList({
  items, total, page, pageSize, isLoading,
  filter, onFilterChange, onPageChange,
}: Props) {
  const totalPages = Math.max(1, Math.ceil(total / pageSize));

  return (
    <div className="space-y-2">
      <Input
        placeholder="Filter..."
        value={filter}
        onChange={(e: React.ChangeEvent<HTMLInputElement>) => onFilterChange(e.target.value)}
        className="h-8 text-xs"
      />

      {isLoading ? (
        <div className="text-xs text-muted-foreground py-4 text-center">Loading...</div>
      ) : items.length === 0 ? (
        <div className="text-xs text-muted-foreground py-4 text-center">No items found</div>
      ) : (
        <div className="space-y-1 max-h-80 overflow-y-auto">
          {items.map((item, i) => (
            <div key={i} className="text-xs p-2 bg-muted rounded">
              {Object.entries(item).map(([key, value]) => (
                <span key={key} className="mr-3">
                  <span className="text-muted-foreground">{key}:</span>{' '}
                  <span className="font-mono">{String(value).slice(0, 80)}</span>
                </span>
              ))}
            </div>
          ))}
        </div>
      )}

      <div className="flex items-center justify-between text-xs">
        <span className="text-muted-foreground">
          {total} total
        </span>
        <div className="flex items-center gap-1">
          <Button
            variant="outline"
            size="sm"
            className="h-6 text-[10px] px-2"
            disabled={page <= 1}
            onClick={() => onPageChange(page - 1)}
          >
            Prev
          </Button>
          <span className="px-2">{page} / {totalPages}</span>
          <Button
            variant="outline"
            size="sm"
            className="h-6 text-[10px] px-2"
            disabled={page >= totalPages}
            onClick={() => onPageChange(page + 1)}
          >
            Next
          </Button>
        </div>
      </div>
    </div>
  );
}
