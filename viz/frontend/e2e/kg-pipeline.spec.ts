import { test, expect } from '@playwright/test';
import { setupConsoleMonitor, checkpoint } from './helpers/checkpoint';

test.describe('KG Pipeline Explorer', () => {
  test('renders pipeline stages and allows navigation', async ({ page }) => {
    setupConsoleMonitor(page);

    await page.goto('/');

    // Switch to KG Pipeline tab
    await page.getByRole('tab', { name: 'KG Pipeline' }).click();
    await checkpoint(page, 'kg-page-loaded');

    // Pipeline card should show loaded stages (not 0/7)
    const pipelineCard = page.getByText(/Pipeline \([1-7]\/7\)/);
    await expect(pipelineCard).toBeVisible({ timeout: 15_000 });

    // Should show stage buttons
    const stage1 = page.locator('button', { hasText: /^1\./ }).first();
    await expect(stage1).toBeVisible({ timeout: 5_000 });

    await checkpoint(page, 'kg-stage-1-chunks');

    // Click on stage 3 (Entities)
    const stage3 = page.locator('button', { hasText: /^3\./ }).first();
    await stage3.click();
    await page.waitForTimeout(1000);
    await checkpoint(page, 'kg-stage-3-entities');

    // Click on stage 6 (Graph)
    const stage6 = page.locator('button', { hasText: /^6\./ }).first();
    await stage6.click();
    await page.waitForTimeout(1000);
    await checkpoint(page, 'kg-stage-6-graph');

    // Data funnel should be visible
    const funnelCard = page.getByText('Data Funnel');
    await expect(funnelCard).toBeVisible();

    await checkpoint(page, 'kg-funnel-visible');
  });

  test('has GraphRAG query input', async ({ page }) => {
    setupConsoleMonitor(page);

    await page.goto('/');

    // Switch to KG Pipeline tab
    await page.getByRole('tab', { name: 'KG Pipeline' }).click();

    // GraphRAG query section should be visible
    const queryInput = page.locator('input[placeholder*="division of labor"]');
    await expect(queryInput).toBeVisible({ timeout: 10_000 });

    // Query button should be present
    const queryButton = page.getByRole('button', { name: 'Query' });
    await expect(queryButton).toBeVisible();

    await checkpoint(page, 'kg-graphrag-ready');
  });
});
