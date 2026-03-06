import { useState, useCallback, useEffect } from 'react';
import {
  EyeIcon,
  ArrowPathIcon,
  MagnifyingGlassIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  QuestionMarkCircleIcon,
  XCircleIcon,
  AdjustmentsHorizontalIcon,
  ChartBarIcon,
} from '@heroicons/react/24/outline';

interface ConfidenceSignal {
  source: string;
  value: number;
  weight: number;
  reasoning: string;
}

interface AnalysisResult {
  success: boolean;
  query: string;
  query_type: string;
  confidence: number;
  confidence_level: string;
  action: string;
  should_verify: boolean;
  verification_query: string | null;
  epistemic_markers: string[];
  signals: ConfidenceSignal[];
  processing_time_ms: number;
  recommendation: string;
}

interface Stats {
  total_queries: number;
  verifications_triggered: number;
  abstentions: number;
  avg_confidence: number;
  verification_rate: number;
  abstention_rate: number;
  query_type_counts: Record<string, number>;
}

export function IntrospectionPanel() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'analyze' | 'stats' | 'settings'>('analyze');

  // Fetch stats
  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch('/api/introspection/stats');
      if (res.ok) {
        const data = await res.json();
        if (data.success) {
          setStats(data);
        }
      }
    } catch (e) {
      console.error('[Introspection] Failed to fetch stats:', e);
    }
  }, []);

  useEffect(() => {
    fetchStats();
    const interval = setInterval(fetchStats, 15000);
    return () => clearInterval(interval);
  }, [fetchStats]);

  const handleAnalyze = async () => {
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch('/api/introspection/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query.trim() }),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const data = await res.json();
      if (data.success) {
        setResult(data);
        fetchStats(); // Refresh stats
      } else {
        setError(data.error || 'Analysis failed');
      }
    } catch (e) {
      console.error('[Introspection] Error:', e);
      setError(e instanceof Error ? e.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const getConfidenceColor = (level: string) => {
    switch (level) {
      case 'high': return 'text-green-400';
      case 'medium': return 'text-yellow-400';
      case 'low': return 'text-orange-400';
      case 'very_low': return 'text-red-400';
      default: return 'text-chat-text-secondary';
    }
  };

  const getConfidenceBg = (level: string) => {
    switch (level) {
      case 'high': return 'bg-green-900/30 border-green-600/30';
      case 'medium': return 'bg-yellow-900/30 border-yellow-600/30';
      case 'low': return 'bg-orange-900/30 border-orange-600/30';
      case 'very_low': return 'bg-red-900/30 border-red-600/30';
      default: return 'bg-chat-sidebar border-chat-border';
    }
  };

  const getActionIcon = (action: string) => {
    switch (action) {
      case 'respond': return <CheckCircleIcon className="w-5 h-5 text-green-400" />;
      case 'respond_hedged': return <ExclamationTriangleIcon className="w-5 h-5 text-yellow-400" />;
      case 'verify': return <MagnifyingGlassIcon className="w-5 h-5 text-orange-400" />;
      case 'abstain': return <XCircleIcon className="w-5 h-5 text-red-400" />;
      default: return <QuestionMarkCircleIcon className="w-5 h-5 text-chat-text-secondary" />;
    }
  };

  const getQueryTypeEmoji = (type: string) => {
    switch (type) {
      case 'factual': return '📚';
      case 'procedural': return '📝';
      case 'analytical': return '🔍';
      case 'opinion': return '💭';
      case 'creative': return '🎨';
      case 'conversational': return '💬';
      default: return '❓';
    }
  };

  return (
    <div className="bg-chat-sidebar rounded-lg p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-chat-text font-medium flex items-center gap-2">
          <EyeIcon className="w-5 h-5 text-cyan-400" />
          Introspection Circuit
        </h3>
        {stats && (
          <div className="text-xs text-chat-text-secondary">
            {stats.total_queries} queries analyzed
          </div>
        )}
      </div>

      {/* Tabs */}
      <div className="flex gap-2 mb-4 border-b border-chat-border pb-2">
        <button
          onClick={() => setActiveTab('analyze')}
          className={`px-3 py-1 rounded text-sm transition-colors ${
            activeTab === 'analyze'
              ? 'bg-cyan-600 text-white'
              : 'text-chat-text-secondary hover:text-chat-text'
          }`}
        >
          Analyze
        </button>
        <button
          onClick={() => setActiveTab('stats')}
          className={`px-3 py-1 rounded text-sm transition-colors ${
            activeTab === 'stats'
              ? 'bg-cyan-600 text-white'
              : 'text-chat-text-secondary hover:text-chat-text'
          }`}
        >
          <ChartBarIcon className="w-4 h-4 inline mr-1" />
          Stats
        </button>
        <button
          onClick={() => setActiveTab('settings')}
          className={`px-3 py-1 rounded text-sm transition-colors ${
            activeTab === 'settings'
              ? 'bg-cyan-600 text-white'
              : 'text-chat-text-secondary hover:text-chat-text'
          }`}
        >
          <AdjustmentsHorizontalIcon className="w-4 h-4 inline mr-1" />
          Settings
        </button>
      </div>

      {/* Analyze Tab */}
      {activeTab === 'analyze' && (
        <div className="space-y-4">
          {/* Input */}
          <div>
            <label className="text-sm text-chat-text-secondary mb-1 block">
              Test Query
            </label>
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Enter a query to analyze uncertainty..."
              className="w-full bg-chat-assistant border border-chat-border rounded p-2 text-chat-text text-sm resize-none h-20 focus:outline-none focus:border-cyan-500"
            />
          </div>

          <button
            onClick={handleAnalyze}
            disabled={loading || !query.trim()}
            className="w-full bg-cyan-600 hover:bg-cyan-700 disabled:bg-cyan-900 disabled:opacity-50 text-white py-2 rounded flex items-center justify-center gap-2 transition-colors"
          >
            {loading ? (
              <>
                <ArrowPathIcon className="w-4 h-4 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <EyeIcon className="w-4 h-4" />
                Analyze Uncertainty
              </>
            )}
          </button>

          {error && (
            <div className="bg-red-900/30 border border-red-600/30 rounded p-3 text-red-400 text-sm">
              Error: {error}
            </div>
          )}

          {/* Result */}
          {result && (
            <div className="space-y-3">
              {/* Confidence Overview */}
              <div className={`rounded border p-3 ${getConfidenceBg(result.confidence_level)}`}>
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    {getActionIcon(result.action)}
                    <span className={`text-lg font-bold ${getConfidenceColor(result.confidence_level)}`}>
                      {(result.confidence * 100).toFixed(0)}%
                    </span>
                    <span className="text-chat-text-secondary text-sm">
                      ({result.confidence_level.replace('_', ' ')})
                    </span>
                  </div>
                  <div className="flex items-center gap-1 text-sm">
                    <span>{getQueryTypeEmoji(result.query_type)}</span>
                    <span className="text-chat-text-secondary">{result.query_type}</span>
                  </div>
                </div>

                {/* Action */}
                <div className="text-sm text-chat-text">
                  <strong>Action:</strong> {result.action.replace('_', ' ')}
                </div>

                {/* Recommendation */}
                <div className="text-xs text-chat-text-secondary mt-1">
                  {result.recommendation}
                </div>
              </div>

              {/* Verification */}
              {result.should_verify && result.verification_query && (
                <div className="bg-orange-900/20 border border-orange-600/30 rounded p-2 text-sm">
                  <div className="flex items-center gap-2 text-orange-400 mb-1">
                    <MagnifyingGlassIcon className="w-4 h-4" />
                    Verification Needed
                  </div>
                  <div className="text-chat-text-secondary text-xs">
                    Suggested search: "{result.verification_query}"
                  </div>
                </div>
              )}

              {/* Epistemic Markers */}
              {result.epistemic_markers.length > 0 && (
                <div className="text-xs text-chat-text-secondary">
                  <strong>Suggested hedges:</strong>
                  <ul className="list-disc list-inside mt-1">
                    {result.epistemic_markers.slice(0, 3).map((marker, i) => (
                      <li key={i}>"{marker}..."</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Signals Breakdown */}
              <div>
                <div className="text-sm text-chat-text-secondary mb-2">
                  Confidence Signals
                </div>
                <div className="space-y-1">
                  {result.signals.map((signal, i) => (
                    <div
                      key={i}
                      className="flex items-center justify-between bg-chat-assistant rounded p-2 text-xs"
                    >
                      <div className="flex items-center gap-2">
                        <span className="text-chat-text font-medium capitalize">
                          {signal.source.replace('_', ' ')}
                        </span>
                        <span className="text-chat-text-secondary">
                          (weight: {signal.weight})
                        </span>
                      </div>
                      <div className={`font-bold ${
                        signal.value >= 0.7 ? 'text-green-400' :
                        signal.value >= 0.4 ? 'text-yellow-400' : 'text-red-400'
                      }`}>
                        {(signal.value * 100).toFixed(0)}%
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Processing time */}
              <div className="text-xs text-chat-text-secondary text-right">
                Processed in {result.processing_time_ms.toFixed(1)}ms
              </div>
            </div>
          )}

          {/* Help text */}
          {!result && !loading && (
            <div className="text-xs text-chat-text-secondary">
              <p className="mb-2">The Introspection Circuit detects uncertainty by:</p>
              <ul className="list-disc list-inside space-y-1">
                <li>Classifying query type (factual, opinion, creative, etc.)</li>
                <li>Extracting verbalized confidence from responses</li>
                <li>Checking consistency across multiple samples</li>
                <li>Integrating with FluxMind & Guardian</li>
              </ul>
            </div>
          )}
        </div>
      )}

      {/* Stats Tab */}
      {activeTab === 'stats' && stats && (
        <div className="space-y-4">
          {/* Overview Stats */}
          <div className="grid grid-cols-2 gap-2">
            <div className="bg-chat-assistant rounded p-3 text-center">
              <div className="text-2xl font-bold text-cyan-400">
                {(stats.avg_confidence * 100).toFixed(0)}%
              </div>
              <div className="text-xs text-chat-text-secondary">Avg Confidence</div>
            </div>
            <div className="bg-chat-assistant rounded p-3 text-center">
              <div className="text-2xl font-bold text-chat-text">
                {stats.total_queries}
              </div>
              <div className="text-xs text-chat-text-secondary">Total Queries</div>
            </div>
            <div className="bg-chat-assistant rounded p-3 text-center">
              <div className="text-2xl font-bold text-orange-400">
                {(stats.verification_rate * 100).toFixed(0)}%
              </div>
              <div className="text-xs text-chat-text-secondary">Verification Rate</div>
            </div>
            <div className="bg-chat-assistant rounded p-3 text-center">
              <div className="text-2xl font-bold text-red-400">
                {(stats.abstention_rate * 100).toFixed(0)}%
              </div>
              <div className="text-xs text-chat-text-secondary">Abstention Rate</div>
            </div>
          </div>

          {/* Query Type Distribution */}
          <div>
            <div className="text-sm text-chat-text-secondary mb-2">
              Query Type Distribution
            </div>
            <div className="space-y-1">
              {Object.entries(stats.query_type_counts).map(([type, count]) => (
                <div key={type} className="flex items-center gap-2">
                  <span className="text-lg">{getQueryTypeEmoji(type)}</span>
                  <span className="text-sm text-chat-text capitalize flex-1">
                    {type}
                  </span>
                  <span className="text-sm text-chat-text-secondary">
                    {count}
                  </span>
                  <div className="w-20 bg-chat-border rounded-full h-2">
                    <div
                      className="bg-cyan-500 h-2 rounded-full"
                      style={{
                        width: `${(count / Math.max(1, stats.total_queries)) * 100}%`
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Settings Tab */}
      {activeTab === 'settings' && (
        <div className="space-y-4 text-sm">
          <div className="text-chat-text-secondary">
            Configuration options for the Introspection Circuit.
          </div>

          <div className="space-y-3">
            <div>
              <label className="text-chat-text-secondary text-xs block mb-1">
                High Confidence Threshold
              </label>
              <input
                type="range"
                min="0.5"
                max="1.0"
                step="0.05"
                defaultValue="0.80"
                className="w-full"
              />
              <div className="flex justify-between text-xs text-chat-text-secondary">
                <span>50%</span>
                <span>80%</span>
                <span>100%</span>
              </div>
            </div>

            <div>
              <label className="text-chat-text-secondary text-xs block mb-1">
                Verify Factual Below
              </label>
              <input
                type="range"
                min="0.3"
                max="0.9"
                step="0.05"
                defaultValue="0.70"
                className="w-full"
              />
              <div className="flex justify-between text-xs text-chat-text-secondary">
                <span>30%</span>
                <span>70%</span>
                <span>90%</span>
              </div>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-chat-text-secondary">Consistency Check</span>
              <input type="checkbox" defaultChecked className="rounded" />
            </div>

            <div className="flex items-center justify-between">
              <span className="text-chat-text-secondary">Auto Verification</span>
              <input type="checkbox" defaultChecked className="rounded" />
            </div>

            <div className="flex items-center justify-between">
              <span className="text-chat-text-secondary">Epistemic Markers</span>
              <input type="checkbox" defaultChecked className="rounded" />
            </div>
          </div>

          <div className="text-xs text-chat-text-secondary pt-2 border-t border-chat-border">
            Backend settings API not yet connected — sliders are display-only.
          </div>
        </div>
      )}
    </div>
  );
}
