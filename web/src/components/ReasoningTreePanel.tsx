import { useState } from 'react';
import {
  ArrowPathIcon,
  BeakerIcon,
  ChevronRightIcon,
  ChevronDownIcon,
  LightBulbIcon,
  CheckCircleIcon,
  QuestionMarkCircleIcon,
} from '@heroicons/react/24/outline';

interface ThoughtNode {
  id: string;
  thought: {
    id: string;
    type: string;
    content: string;
    confidence: number;
  };
  depth: number;
  visits: number;
  value: number;
  ucb1: number | null;
  state: string;
  is_terminal: boolean;
  is_successful: boolean;
  children_count: number;
  children: ThoughtNode[];
}

interface ReasoningStep {
  type: string;
  content: string;
  confidence: number;
  value: number;
}

interface ReasoningResult {
  success: boolean;
  session_id: string;
  answer: string;
  confidence: number;
  reasoning_steps: ReasoningStep[];
  summary: string;
  metadata: {
    iterations: number;
    nodes_explored: number;
    time_taken: number;
    reflections_count: number;
  };
}

interface TreeVisualization {
  success: boolean;
  tree: ThoughtNode;
  stats: {
    iterations: number;
    nodes: number;
    reflections: number;
    max_depth: number;
  };
}

// Tree node component
function TreeNode({ node, depth = 0, expanded, onToggle }: {
  node: ThoughtNode;
  depth?: number;
  expanded: Set<string>;
  onToggle: (id: string) => void;
}) {
  const isExpanded = expanded.has(node.id);
  const hasChildren = node.children && node.children.length > 0;

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'root':
        return '🎯';
      case 'reasoning':
        return '🧠';
      case 'action':
        return '⚡';
      case 'conclusion':
        return '✅';
      case 'observation':
        return '👁️';
      case 'reflection':
        return '🪞';
      default:
        return '💭';
    }
  };

  const getValueColor = (value: number) => {
    if (value >= 0.7) return 'text-green-400';
    if (value >= 0.4) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getStateColor = (state: string) => {
    switch (state) {
      case 'evaluated':
        return 'bg-blue-900/30 border-blue-600/30';
      case 'terminal':
        return 'bg-green-900/30 border-green-600/30';
      case 'pruned':
        return 'bg-gray-900/30 border-gray-600/30';
      default:
        return 'bg-chat-sidebar border-chat-border';
    }
  };

  return (
    <div className="ml-2">
      <div
        className={`flex items-start gap-2 p-2 rounded border ${getStateColor(node.state)} cursor-pointer hover:bg-opacity-50 transition-colors`}
        onClick={() => hasChildren && onToggle(node.id)}
      >
        {/* Expand/collapse button */}
        <div className="w-5 h-5 flex items-center justify-center">
          {hasChildren ? (
            isExpanded ? (
              <ChevronDownIcon className="w-4 h-4 text-chat-text-secondary" />
            ) : (
              <ChevronRightIcon className="w-4 h-4 text-chat-text-secondary" />
            )
          ) : (
            <div className="w-4 h-4" />
          )}
        </div>

        {/* Type icon */}
        <span className="text-lg">{getTypeIcon(node.thought.type)}</span>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="text-chat-text text-sm truncate">
            {node.thought.content.length > 100
              ? node.thought.content.substring(0, 100) + '...'
              : node.thought.content}
          </div>
          <div className="flex items-center gap-3 mt-1 text-xs text-chat-text-secondary">
            <span className={getValueColor(node.value)}>
              Value: {(node.value * 100).toFixed(0)}%
            </span>
            <span>Visits: {node.visits}</span>
            {node.ucb1 !== null && (
              <span>UCB: {node.ucb1.toFixed(2)}</span>
            )}
            {node.is_terminal && (
              <span className={node.is_successful ? 'text-green-400' : 'text-red-400'}>
                {node.is_successful ? '✓ Success' : '✗ Failed'}
              </span>
            )}
          </div>
        </div>

        {/* Children count badge */}
        {hasChildren && (
          <div className="px-2 py-0.5 bg-chat-assistant rounded text-xs text-chat-text-secondary">
            {node.children_count}
          </div>
        )}
      </div>

      {/* Render children */}
      {hasChildren && isExpanded && (
        <div className="ml-4 mt-1 border-l border-chat-border pl-2">
          {node.children.map((child) => (
            <TreeNode
              key={child.id}
              node={child}
              depth={depth + 1}
              expanded={expanded}
              onToggle={onToggle}
            />
          ))}
        </div>
      )}
    </div>
  );
}

// Generate a plain-English explanation from MCTS result data (no LLM call needed)
function generateExplanation(result: ReasoningResult, tree: TreeVisualization | null): string {
  if (!result) return '';
  const { metadata, reasoning_steps, confidence } = result;
  const nodes = metadata?.nodes_explored || 0;
  const iters = metadata?.iterations || 0;
  const reflections = metadata?.reflections_count || 0;
  const timeTaken = metadata?.time_taken?.toFixed(1) || '?';

  const stepTypes = reasoning_steps?.reduce((acc: Record<string, number>, s) => {
    acc[s.type] = (acc[s.type] || 0) + 1;
    return acc;
  }, {}) || {};

  const typeDescriptions: Record<string, string> = {
    reasoning: 'analytical reasoning branches',
    action: 'action steps',
    conclusion: 'conclusions',
    observation: 'observations',
    reflection: 'self-corrections',
  };

  const stepSummary = Object.entries(stepTypes)
    .map(([type, count]) => `${count} ${typeDescriptions[type] || type}`)
    .join(', ');

  const confidenceLabel = confidence >= 0.8 ? 'high' : confidence >= 0.5 ? 'moderate' : 'low';
  const reflectionNote = reflections > 0
    ? ` Along the way, ${reflections} self-correction${reflections > 1 ? 's' : ''} helped prune poor paths and refocus the search.`
    : '';

  const maxDepth = tree?.stats?.max_depth || '?';

  return [
    `MCTS explored ${nodes} thought nodes across ${iters} iterations in ${timeTaken}s.`,
    '',
    stepSummary ? `The winning path consists of: ${stepSummary}.` : '',
    '',
    `Search depth reached ${maxDepth} levels.${reflectionNote}`,
    '',
    `Final confidence: ${(confidence * 100).toFixed(0)}% (${confidenceLabel}).`,
    confidence < 0.5 ? 'The low confidence suggests the problem may need more iterations or a different framing.' : '',
  ].filter(Boolean).join('\n');
}

function generateMarkdown(problem: string, result: ReasoningResult, tree: TreeVisualization | null): string {
  const lines = [
    `# MCTS Reasoning: ${problem}`,
    '',
    `**Confidence:** ${(result.confidence * 100).toFixed(0)}%  `,
    `**Nodes explored:** ${result.metadata?.nodes_explored || 0}  `,
    `**Iterations:** ${result.metadata?.iterations || 0}  `,
    `**Time:** ${result.metadata?.time_taken?.toFixed(1) || '?'}s`,
    '',
    '## Answer',
    '',
    result.answer,
    '',
    '## Reasoning Path',
    '',
    ...(result.reasoning_steps || []).map((s, i) =>
      `**Step ${i + 1}** [${s.type}] (${(s.value * 100).toFixed(0)}%)\n${s.content}`
    ),
    '',
    '## Explanation',
    '',
    generateExplanation(result, tree),
  ];
  return lines.join('\n');
}

export function ReasoningTreePanel() {
  const [problem, setProblem] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ReasoningResult | null>(null);
  const [tree, setTree] = useState<TreeVisualization | null>(null);
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set());
  const [activeTab, setActiveTab] = useState<'input' | 'result' | 'tree' | 'explain'>('input');
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const handleThinkDeeply = async () => {
    if (!problem.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);
    setTree(null);

    try {
      const res = await fetch('/api/reasoning-tree/think', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          problem: problem.trim(),
          max_iterations: 30,
          max_depth: 10,
          branching_factor: 5,
        }),
      });

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }

      const data = await res.json();
      setResult(data);

      // Fetch tree visualization
      if (data.session_id) {
        const treeRes = await fetch(`/api/reasoning-tree/tree/${data.session_id}`);
        if (treeRes.ok) {
          const treeData = await treeRes.json();
          setTree(treeData);
          // Expand root by default
          if (treeData.tree?.id) {
            setExpandedNodes(new Set([treeData.tree.id]));
          }
        }
      }

      setActiveTab('result');
    } catch (e) {
      console.error('[ReasoningTree] Error:', e);
      setError(e instanceof Error ? e.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const toggleNode = (id: string) => {
    setExpandedNodes((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const expandAll = () => {
    if (!tree?.tree) return;
    const allIds = new Set<string>();
    const collect = (node: ThoughtNode) => {
      allIds.add(node.id);
      node.children?.forEach(collect);
    };
    collect(tree.tree);
    setExpandedNodes(allIds);
  };

  const collapseAll = () => {
    if (!tree?.tree) return;
    setExpandedNodes(new Set([tree.tree.id]));
  };

  return (
    <div className="bg-chat-sidebar rounded-lg p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-chat-text font-medium flex items-center gap-2">
          <BeakerIcon className="w-5 h-5 text-purple-400" />
          MCTS Reasoning Tree
        </h3>
        {result && (
          <div className={`px-2 py-1 rounded text-xs ${
            result.success ? 'bg-green-900/30 text-green-400' : 'bg-red-900/30 text-red-400'
          }`}>
            {result.success ? 'Success' : 'Failed'}
          </div>
        )}
      </div>

      {/* Tabs */}
      <div className="flex gap-2 mb-4 border-b border-chat-border pb-2">
        <button
          onClick={() => setActiveTab('input')}
          className={`px-3 py-1 rounded text-sm transition-colors ${
            activeTab === 'input'
              ? 'bg-purple-600 text-white'
              : 'text-chat-text-secondary hover:text-chat-text'
          }`}
        >
          Input
        </button>
        <button
          onClick={() => setActiveTab('result')}
          disabled={!result}
          className={`px-3 py-1 rounded text-sm transition-colors ${
            activeTab === 'result'
              ? 'bg-purple-600 text-white'
              : !result
                ? 'text-chat-text-secondary/40 cursor-not-allowed'
                : 'text-chat-text-secondary hover:text-chat-text hover:bg-chat-assistant'
          }`}
          title={!result ? 'Run a reasoning session first' : 'View results'}
        >
          Result {!result && '🔒'}
        </button>
        <button
          onClick={() => setActiveTab('tree')}
          disabled={!tree}
          className={`px-3 py-1 rounded text-sm transition-colors ${
            activeTab === 'tree'
              ? 'bg-purple-600 text-white'
              : !tree
                ? 'text-chat-text-secondary/40 cursor-not-allowed'
                : 'text-chat-text-secondary hover:text-chat-text hover:bg-chat-assistant'
          }`}
          title={!tree ? 'Run a reasoning session first' : 'View reasoning tree'}
        >
          Tree View {!tree && '🔒'}
        </button>
        <button
          onClick={() => setActiveTab('explain')}
          disabled={!result}
          className={`px-3 py-1 rounded text-sm transition-colors ${
            activeTab === 'explain'
              ? 'bg-purple-600 text-white'
              : !result
                ? 'text-chat-text-secondary/40 cursor-not-allowed'
                : 'text-chat-text-secondary hover:text-chat-text hover:bg-chat-assistant'
          }`}
          title={!result ? 'Run a reasoning session first' : 'Plain-English explanation'}
        >
          Explain {!result && '🔒'}
        </button>
      </div>

      {/* Input Tab */}
      {activeTab === 'input' && (
        <div className="space-y-4">
          <div>
            <label className="text-sm text-chat-text-secondary mb-1 block">
              Problem / Question
            </label>
            <textarea
              value={problem}
              onChange={(e) => setProblem(e.target.value)}
              placeholder="Enter a complex problem for deep reasoning..."
              className="w-full bg-chat-assistant border border-chat-border rounded p-2 text-chat-text text-sm resize-none h-24 focus:outline-none focus:border-purple-500"
            />
          </div>

          <button
            onClick={handleThinkDeeply}
            disabled={loading || !problem.trim()}
            className="w-full bg-purple-600 hover:bg-purple-700 disabled:bg-purple-900 disabled:opacity-50 text-white py-2 rounded flex items-center justify-center gap-2 transition-colors"
          >
            {loading ? (
              <>
                <ArrowPathIcon className="w-4 h-4 animate-spin" />
                Reasoning...
              </>
            ) : (
              <>
                <LightBulbIcon className="w-4 h-4" />
                Think Deeply
              </>
            )}
          </button>

          {error && (
            <div className="bg-red-900/30 border border-red-600/30 rounded p-3 text-red-400 text-sm">
              Error: {error}
            </div>
          )}

          <div className="text-xs text-chat-text-secondary">
            <p className="mb-2">MCTS explores multiple reasoning paths simultaneously:</p>
            <ul className="list-disc list-inside space-y-1">
              <li>Generates candidate thoughts (branching)</li>
              <li>Evaluates each path's quality</li>
              <li>Backpropagates values through the tree</li>
              <li>Reflects on failures to improve search</li>
            </ul>
          </div>
        </div>
      )}

      {/* Result Tab */}
      {activeTab === 'result' && result && (
        <div className="space-y-4">
          {/* Stats */}
          <div className="grid grid-cols-4 gap-2">
            <div className="bg-chat-assistant rounded p-2 text-center">
              <div className="text-lg font-bold text-purple-400">
                {(result.confidence * 100).toFixed(0)}%
              </div>
              <div className="text-xs text-chat-text-secondary">Confidence</div>
            </div>
            <div className="bg-chat-assistant rounded p-2 text-center">
              <div className="text-lg font-bold text-chat-text">
                {result.metadata?.iterations || 0}
              </div>
              <div className="text-xs text-chat-text-secondary">Iterations</div>
            </div>
            <div className="bg-chat-assistant rounded p-2 text-center">
              <div className="text-lg font-bold text-chat-text">
                {result.metadata?.nodes_explored || 0}
              </div>
              <div className="text-xs text-chat-text-secondary">Nodes</div>
            </div>
            <div className="bg-chat-assistant rounded p-2 text-center">
              <div className="text-lg font-bold text-chat-text">
                {result.metadata?.time_taken?.toFixed(1) || 0}s
              </div>
              <div className="text-xs text-chat-text-secondary">Time</div>
            </div>
          </div>

          {/* Answer */}
          <div>
            <div className="text-sm text-chat-text-secondary mb-1 flex items-center gap-1">
              <CheckCircleIcon className="w-4 h-4 text-green-400" />
              Answer
            </div>
            <div className="bg-chat-assistant rounded p-3 text-chat-text text-sm">
              {result.answer}
            </div>
          </div>

          {/* Reasoning Steps */}
          {result.reasoning_steps && result.reasoning_steps.length > 0 && (
            <div>
              <div className="text-sm text-chat-text-secondary mb-2">
                Reasoning Path ({result.reasoning_steps.length} steps)
              </div>
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {result.reasoning_steps.map((step, i) => (
                  <div
                    key={i}
                    className="bg-chat-assistant rounded p-2 text-sm border-l-2 border-purple-500"
                  >
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-purple-400 font-medium">Step {i + 1}</span>
                      <span className="text-chat-text-secondary text-xs">
                        [{step.type}]
                      </span>
                      <span className="text-green-400 text-xs ml-auto">
                        {(step.value * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="text-chat-text">{step.content}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Reflections count */}
          {result.metadata?.reflections_count > 0 && (
            <div className="text-xs text-chat-text-secondary">
              💡 {result.metadata.reflections_count} reflection(s) generated during search
            </div>
          )}
        </div>
      )}

      {/* Tree Tab */}
      {activeTab === 'tree' && tree && tree.tree && (
        <div className="space-y-4">
          {/* Tree controls */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4 text-xs text-chat-text-secondary">
              <span>Nodes: {tree.stats?.nodes || 0}</span>
              <span>Max Depth: {tree.stats?.max_depth || 0}</span>
              <span>Reflections: {tree.stats?.reflections || 0}</span>
            </div>
            <div className="flex gap-2">
              <button
                onClick={expandAll}
                className="px-2 py-1 text-xs bg-chat-assistant rounded hover:bg-chat-border transition-colors text-chat-text-secondary"
              >
                Expand All
              </button>
              <button
                onClick={collapseAll}
                className="px-2 py-1 text-xs bg-chat-assistant rounded hover:bg-chat-border transition-colors text-chat-text-secondary"
              >
                Collapse
              </button>
            </div>
          </div>

          {/* Tree visualization */}
          <div className="bg-chat-assistant rounded p-2 max-h-96 overflow-y-auto">
            <TreeNode
              node={tree.tree}
              expanded={expandedNodes}
              onToggle={toggleNode}
            />
          </div>

          {/* Legend */}
          <div className="flex flex-wrap gap-3 text-xs text-chat-text-secondary">
            <span>🎯 Root</span>
            <span>🧠 Reasoning</span>
            <span>⚡ Action</span>
            <span>✅ Conclusion</span>
            <span>🪞 Reflection</span>
          </div>
        </div>
      )}

      {/* Explain Tab */}
      {activeTab === 'explain' && result && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h4 className="text-sm text-chat-text-secondary">Plain-English Explanation</h4>
            <button
              onClick={() => {
                const md = generateMarkdown(problem, result, tree);
                navigator.clipboard.writeText(md).then(() => {
                  setCopied(true);
                  setTimeout(() => setCopied(false), 2000);
                });
              }}
              className="px-2 py-1 text-xs bg-chat-assistant hover:bg-chat-border rounded text-chat-text-secondary hover:text-chat-text transition-colors"
            >
              {copied ? '✓ Copied!' : 'Export Markdown'}
            </button>
          </div>

          {/* Explanation */}
          <div className="bg-chat-assistant rounded p-3 text-sm text-chat-text leading-relaxed whitespace-pre-line">
            {generateExplanation(result, tree)}
          </div>

          {/* Visual confidence bar */}
          <div>
            <div className="flex justify-between text-xs text-chat-text-secondary mb-1">
              <span>Confidence</span>
              <span>{(result.confidence * 100).toFixed(0)}%</span>
            </div>
            <div className="bg-chat-assistant rounded-full h-2">
              <div
                className={`h-2 rounded-full transition-all ${
                  result.confidence >= 0.7 ? 'bg-green-500' :
                  result.confidence >= 0.4 ? 'bg-yellow-500' : 'bg-red-500'
                }`}
                style={{ width: `${Math.min(result.confidence * 100, 100)}%` }}
              />
            </div>
          </div>

          {/* Step type breakdown */}
          {result.reasoning_steps?.length > 0 && (
            <div>
              <div className="text-xs text-chat-text-secondary mb-2">Reasoning composition</div>
              <div className="flex flex-wrap gap-2">
                {Object.entries(
                  result.reasoning_steps.reduce((acc: Record<string, number>, s) => {
                    acc[s.type] = (acc[s.type] || 0) + 1;
                    return acc;
                  }, {})
                ).map(([type, count]) => (
                  <div key={type} className="bg-chat-assistant rounded px-2 py-1 text-xs">
                    <span className="text-purple-400">{count}×</span>
                    <span className="text-chat-text-secondary ml-1">{type}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Best path highlight */}
          {result.reasoning_steps && result.reasoning_steps.length > 0 && (
            <div>
              <div className="text-xs text-chat-text-secondary mb-2">Best step (highest value)</div>
              <div className="bg-green-900/20 border border-green-600/20 rounded p-2 text-sm">
                <div className="text-green-400 text-xs mb-1">
                  {(Math.max(...result.reasoning_steps.map((s) => s.value)) * 100).toFixed(0)}% confidence
                </div>
                <div className="text-chat-text">
                  {result.reasoning_steps.reduce((best, s) => s.value > best.value ? s : best).content}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Empty state for result/tree tabs */}
      {activeTab === 'result' && !result && !loading && (
        <div className="text-center py-8 text-chat-text-secondary">
          <QuestionMarkCircleIcon className="w-12 h-12 mx-auto mb-2 opacity-50" />
          <p>No reasoning result yet</p>
          <p className="text-xs mt-1">Enter a problem and click "Think Deeply"</p>
        </div>
      )}

      {activeTab === 'tree' && !tree && !loading && (
        <div className="text-center py-8 text-chat-text-secondary">
          <QuestionMarkCircleIcon className="w-12 h-12 mx-auto mb-2 opacity-50" />
          <p>No reasoning tree available</p>
          <p className="text-xs mt-1">Run a deep reasoning session first</p>
        </div>
      )}
    </div>
  );
}
