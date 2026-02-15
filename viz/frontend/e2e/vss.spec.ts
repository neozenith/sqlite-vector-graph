import { test, expect } from '@playwright/test';
import { setupConsoleMonitor, checkpoint } from './helpers/checkpoint';

test.describe('VSS Explorer', () => {
  test('renders embeddings tab and discovers indexes', async ({ page }) => {
    setupConsoleMonitor(page);

    await page.goto('/');
    await checkpoint(page, 'vss-page-loaded');

    // The "Embeddings" tab should be active by default
    const tab = page.getByRole('tab', { name: 'Embeddings' });
    await expect(tab).toHaveAttribute('data-state', 'active');

    // Wait for the index selector to populate
    const indexSelect = page.locator('select').first();
    await expect(indexSelect).toBeVisible({ timeout: 10_000 });

    // Wait for at least one real index to appear in the dropdown
    await expect(async () => {
      const options = await indexSelect.locator('option').allTextContents();
      const realOptions = options.filter((o) => !o.includes('Select'));
      expect(realOptions.length).toBeGreaterThanOrEqual(1);
    }).toPass({ timeout: 10_000 });

    await checkpoint(page, 'vss-index-selected');

    // UMAP projection can take > 30s for 1850×384-dim vectors.
    // Verify we see either "Projecting..." or the stats card.
    const projecting = page.locator('text=Projecting embeddings');
    const stats = page.locator('text=Points:');

    await expect(projecting.or(stats)).toBeVisible({ timeout: 5_000 });
    await checkpoint(page, 'vss-umap-started');
  });
});
