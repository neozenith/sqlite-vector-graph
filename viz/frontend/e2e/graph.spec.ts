import { test, expect } from '@playwright/test';
import { setupConsoleMonitor, checkpoint } from './helpers/checkpoint';

test.describe('Graph Explorer', () => {
  test('renders graph page and loads network', async ({ page }) => {
    setupConsoleMonitor(page);

    await page.goto('/');

    // Click the "Graph" sidebar link to navigate
    await page.getByRole('link', { name: 'Graph' }).click();
    await page.waitForURL(/\/graph/);
    await checkpoint(page, 'graph-page-loaded');

    // The dataset picker shows "Edge Tables" heading when no dataset selected
    await expect(page.getByRole('heading', { name: 'Edge Tables' })).toBeVisible({ timeout: 10_000 });

    // Wait for at least one edge table card to appear (contains "edges" badge)
    await expect(async () => {
      const cards = await page.locator('text=/\\d+ edges/').count();
      expect(cards).toBeGreaterThanOrEqual(1);
    }).toPass({ timeout: 10_000 });

    await checkpoint(page, 'graph-tables-discovered');

    // Click the first edge table card to select it
    const firstCard = page.locator('[class*="cursor-pointer"]').first();
    await firstCard.click();
    await page.waitForURL(/\/graph\/.+/);

    // Graph stats should show (indicates data loaded)
    const nodesText = page.locator('text=Nodes:');
    await expect(nodesText).toBeVisible({ timeout: 15_000 });

    await checkpoint(page, 'graph-network-rendered');

    // Centrality selector should be present in the sidebar
    const centralitySelect = page.locator('select').first();
    await expect(centralitySelect).toBeVisible();

    await checkpoint(page, 'graph-centrality-visible');
  });
});
