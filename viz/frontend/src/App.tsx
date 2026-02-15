import { QueryClient, QueryClientProvider, useQuery } from '@tanstack/react-query';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { useAppStore } from '@/stores/app-store';
import { useDeepLink } from '@/hooks/useDeepLink';
import { VSSExplorer } from '@/components/VSSExplorer';
import { GraphExplorer } from '@/components/GraphExplorer';
import { KGPipelineExplorer } from '@/components/KGPipelineExplorer';
import { fetchJSON } from '@/lib/services/api-client';
import type { HealthStatus, TabId } from '@/lib/types';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 60_000,
      retry: 1,
    },
  },
});

function AppContent() {
  useDeepLink();
  const { activeTab, setActiveTab } = useAppStore();

  const { data: health } = useQuery({
    queryKey: ['health'],
    queryFn: () => fetchJSON<HealthStatus>('/api/health'),
    staleTime: 30_000,
  });

  return (
    <div className="flex flex-col h-screen">
      {/* Header */}
      <header className="border-b px-4 py-2 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-3">
          <h1 className="text-lg font-semibold tracking-tight">muninn viz</h1>
          {health?.extension_loaded && (
            <Badge variant="secondary" className="text-[10px]">extension loaded</Badge>
          )}
        </div>
        {health && (
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <span>{health.db_path.split('/').pop()}</span>
            {health.hnsw_index_count != null && (
              <Badge variant="outline" className="text-[10px]">
                {health.hnsw_index_count} indexes
              </Badge>
            )}
            {health.edge_table_count != null && (
              <Badge variant="outline" className="text-[10px]">
                {health.edge_table_count} graphs
              </Badge>
            )}
          </div>
        )}
      </header>

      {/* Tab Navigation + Content */}
      <Tabs
        value={activeTab}
        onValueChange={(v) => setActiveTab(v as TabId)}
        className="flex-1 flex flex-col overflow-hidden"
      >
        <div className="border-b px-4">
          <TabsList className="h-9">
            <TabsTrigger value="vss" className="text-xs">Embeddings</TabsTrigger>
            <TabsTrigger value="graph" className="text-xs">Graph</TabsTrigger>
            <TabsTrigger value="kg" className="text-xs">KG Pipeline</TabsTrigger>
          </TabsList>
        </div>

        <div className="flex-1 overflow-hidden p-4">
          <TabsContent value="vss" className="h-full mt-0">
            <VSSExplorer />
          </TabsContent>
          <TabsContent value="graph" className="h-full mt-0">
            <GraphExplorer />
          </TabsContent>
          <TabsContent value="kg" className="h-full mt-0">
            <KGPipelineExplorer />
          </TabsContent>
        </div>
      </Tabs>
    </div>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AppContent />
    </QueryClientProvider>
  );
}
