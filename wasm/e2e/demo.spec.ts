/**
 * E2E tests for the muninn WASM demo.
 *
 * Tests the full flow: page load → library init → DB load → search → visualization.
 * Timeouts are generous to accommodate first-time CDN downloads and model loading.
 */
import { test, expect } from '@playwright/test';
import { setupConsoleMonitor, checkpoint } from './helpers/checkpoint';

test.describe('muninn WASM Demo', () => {
  test.setTimeout(120_000); // 2 minutes for CDN + model downloads

  test('loads and searches the knowledge graph', async ({ page }) => {
    setupConsoleMonitor(page);

    // 1. Navigate to the demo page
    await page.goto('/');
    await expect(page.locator('h1')).toContainText('muninn');
    await checkpoint(page, '01-page-loaded');

    // 2. Wait for WASM module to initialize
    await expect(page.locator('#status-wasm[data-status="ready"]')).toBeVisible({
      timeout: 30_000,
    });
    await checkpoint(page, '02-wasm-ready');

    // 3. Verify database was loaded (footer shows schema info)
    await expect(page.locator('#db-status')).toContainText('chunks', {
      timeout: 10_000,
    });
    await checkpoint(page, '03-database-loaded');

    // 4. Wait for libraries to load
    //    Deck.GL and Cytoscape should be quick (CDN scripts)
    await expect(
      page.locator('#status-deckgl[data-status="ready"], #status-deckgl[data-status="error"]')
    ).toBeVisible({ timeout: 10_000 });
    await expect(
      page.locator('#status-cytoscape[data-status="ready"], #status-cytoscape[data-status="error"]')
    ).toBeVisible({ timeout: 10_000 });
    await checkpoint(page, '04-viz-libraries-loaded');

    // 5. Wait for Transformers.js model (first download can be 10-30s)
    await expect(page.locator('#status-transformers[data-status="ready"]')).toBeVisible({
      timeout: 60_000,
    });
    await checkpoint(page, '05-transformers-ready');

    // 6. Search input should now be enabled
    const searchInput = page.locator('#search-input');
    await expect(searchInput).toBeEnabled({ timeout: 5_000 });
    await checkpoint(page, '06-search-enabled');

    // 7. Perform a search
    await searchInput.fill('trade and commerce between nations');
    // Wait for debounce (300ms) + embedding generation + search
    await expect(page.locator('#results-panel')).toBeVisible({ timeout: 30_000 });
    await checkpoint(page, '07-search-results');

    // 8. Verify results appeared
    const resultCards = page.locator('.result-card');
    await expect(resultCards.first()).toBeVisible({ timeout: 5_000 });
    const resultCount = await resultCards.count();
    expect(resultCount).toBeGreaterThan(0);
    await checkpoint(page, '08-results-verified');

    // 9. Verify Cytoscape graph has nodes (graph traversal from search)
    const cytoscapeCount = page.locator('#cytoscape-count');
    await expect(cytoscapeCount).not.toHaveText('0 nodes', { timeout: 10_000 });
    await checkpoint(page, '09-graph-populated');

    // 10. Verify Deck.GL point count updated
    const deckglCount = page.locator('#deckgl-count');
    await expect(deckglCount).not.toHaveText('0 points', { timeout: 10_000 });
    await checkpoint(page, '10-embeddings-visualized');
  });
});
