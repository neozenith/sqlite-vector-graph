import { test, expect } from '@playwright/test';
import { setupConsoleMonitor, checkpoint } from './helpers/checkpoint';

test.describe('VSS Explorer', () => {
  test('renders embeddings page and discovers indexes', async ({ page }) => {
    setupConsoleMonitor(page);

    // Root redirects to /embeddings/ (the default sidebar route)
    await page.goto('/');
    await checkpoint(page, 'vss-page-loaded');

    // The sidebar "Embeddings" link should be the active route
    const sidebarLink = page.getByRole('link', { name: 'Embeddings' });
    await expect(sidebarLink).toBeVisible();

    // The dataset picker shows "HNSW Indexes" heading when no dataset selected
    await expect(page.getByText('HNSW Indexes')).toBeVisible({ timeout: 10_000 });

    // Wait for at least one index card to appear (contains "points" badge)
    await expect(async () => {
      const cards = await page.locator('text=/\\d+ points/').count();
      expect(cards).toBeGreaterThanOrEqual(1);
    }).toPass({ timeout: 10_000 });

    await checkpoint(page, 'vss-indexes-discovered');

    // Click the first index card to select it
    const firstCard = page.locator('[class*="cursor-pointer"]').first();
    await firstCard.click();
    await page.waitForURL(/\/embeddings\/.+/);

    await checkpoint(page, 'vss-index-selected');

    // UMAP projection can take > 30s for large vectors.
    // Verify we see either "Projecting..." or the stats card.
    const projecting = page.locator('text=Projecting embeddings');
    const stats = page.locator('text=Points:');

    await expect(projecting.or(stats)).toBeVisible({ timeout: 5_000 });
    await checkpoint(page, 'vss-umap-started');
  });
});
