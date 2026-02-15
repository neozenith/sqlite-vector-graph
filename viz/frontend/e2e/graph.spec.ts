import { test, expect } from '@playwright/test';
import { setupConsoleMonitor, checkpoint } from './helpers/checkpoint';

test.describe('Graph Explorer', () => {
  test('renders graph tab and loads network', async ({ page }) => {
    setupConsoleMonitor(page);

    await page.goto('/');

    // Switch to Graph tab
    await page.getByRole('tab', { name: 'Graph' }).click();
    await checkpoint(page, 'graph-page-loaded');

    // Wait for edge table selector to populate with real options
    const tableSelect = page.locator('select').first();
    await expect(tableSelect).toBeVisible({ timeout: 10_000 });

    await expect(async () => {
      const options = await tableSelect.locator('option').allTextContents();
      const realOptions = options.filter((o) => !o.includes('Select'));
      expect(realOptions.length).toBeGreaterThanOrEqual(1);
    }).toPass({ timeout: 10_000 });

    await checkpoint(page, 'graph-table-selected');

    // Graph stats should show (indicates data loaded)
    const nodesText = page.locator('text=Nodes:');
    await expect(nodesText).toBeVisible({ timeout: 15_000 });

    await checkpoint(page, 'graph-network-rendered');

    // Centrality selector should be present
    const centralitySelect = page.locator('select').nth(1);
    await expect(centralitySelect).toBeVisible();

    await checkpoint(page, 'graph-centrality-visible');
  });
});
