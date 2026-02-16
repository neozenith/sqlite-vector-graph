import { test, expect } from '@playwright/test';
import { setupConsoleMonitor, checkpoint } from './helpers/checkpoint';

test.describe('KG Pipeline Explorer', () => {
  test('renders pipeline stages and allows navigation', async ({ page }) => {
    setupConsoleMonitor(page);

    await page.goto('/');

    // Click the "KG Pipeline" sidebar link
    await page.getByRole('link', { name: 'KG Pipeline' }).click();
    await page.waitForURL(/\/kg\/?$/);
    await checkpoint(page, 'kg-page-loaded');

    // Pipeline heading should show loaded stages (not 0/7)
    const pipelineHeading = page.getByText(/KG Pipeline \([1-7]\/7\)/);
    await expect(pipelineHeading).toBeVisible({ timeout: 15_000 });

    // Stage cards are rendered as Links with "N. Name" text
    const stage1 = page.locator('a', { hasText: /^1\./ }).first();
    await expect(stage1).toBeVisible({ timeout: 5_000 });

    await checkpoint(page, 'kg-overview-loaded');

    // Click on stage 3 (Entity Extraction) card — navigates to stage page
    const stage3 = page.locator('a', { hasText: /^3\./ }).first();
    await stage3.click();
    await page.waitForURL(/\/kg\/entity/);

    // KG stage pills should be visible on stage pages (exact: true avoids matching stage card links)
    const entityPill = page.getByRole('link', { name: 'Entity Extraction', exact: true });
    await expect(entityPill).toBeVisible({ timeout: 5_000 });

    await checkpoint(page, 'kg-stage-3-entities');

    // Navigate back to overview via pill
    const overviewPill = page.getByRole('link', { name: 'Overview' });
    await overviewPill.click();
    await page.waitForURL(/\/kg\/?$/);

    // Data funnel should be visible on overview
    const funnelCard = page.getByText('Data Funnel');
    await expect(funnelCard).toBeVisible({ timeout: 10_000 });

    await checkpoint(page, 'kg-funnel-visible');
  });

  test('has GraphRAG query page', async ({ page }) => {
    setupConsoleMonitor(page);

    // Navigate directly to the KG query page
    await page.goto('/kg/query/');
    await checkpoint(page, 'kg-query-page-loaded');

    // Search input should be visible
    const queryInput = page.locator('input[placeholder*="knowledge graph"]');
    await expect(queryInput).toBeVisible({ timeout: 10_000 });

    // Search button should be present
    const searchButton = page.getByRole('button', { name: 'Search' });
    await expect(searchButton).toBeVisible();

    await checkpoint(page, 'kg-graphrag-ready');
  });
});
