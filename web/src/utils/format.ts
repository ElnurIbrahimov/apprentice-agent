/**
 * Utility functions for consistent formatting across the app.
 */

/**
 * Format a decimal (0-1) as a percentage string.
 * @param value - Decimal value between 0 and 1
 * @param decimals - Number of decimal places (default: 1)
 * @returns Formatted percentage string like "54.4%"
 */
export function formatPercent(value: number, decimals: number = 1): string {
  if (typeof value !== 'number' || isNaN(value)) {
    return '0%';
  }
  return `${(value * 100).toFixed(decimals)}%`;
}

/**
 * Format a number with a specific number of decimal places.
 * @param value - Number to format
 * @param decimals - Number of decimal places (default: 2)
 * @returns Formatted number string
 */
export function formatNumber(value: number, decimals: number = 2): string {
  if (typeof value !== 'number' || isNaN(value)) {
    return '0';
  }
  return value.toFixed(decimals);
}

/**
 * Format a confidence value (0-100) as a percentage.
 * @param value - Confidence value (0-100)
 * @returns Formatted percentage string like "85%"
 */
export function formatConfidence(value: number): string {
  if (typeof value !== 'number' || isNaN(value)) {
    return '0%';
  }
  return `${Math.round(value)}%`;
}

/**
 * Format a timestamp as a relative time string.
 * @param timestamp - ISO timestamp string or Date
 * @returns Relative time string like "2 minutes ago"
 */
export function formatRelativeTime(timestamp: string | Date): string {
  const date = typeof timestamp === 'string' ? new Date(timestamp) : timestamp;
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffSecs = Math.floor(diffMs / 1000);
  const diffMins = Math.floor(diffSecs / 60);
  const diffHours = Math.floor(diffMins / 60);
  const diffDays = Math.floor(diffHours / 24);

  if (diffSecs < 60) return 'just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;

  return date.toLocaleDateString();
}
