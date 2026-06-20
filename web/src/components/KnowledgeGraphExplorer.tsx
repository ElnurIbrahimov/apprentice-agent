import { useState, useEffect, useRef, useCallback } from 'react';
import { MagnifyingGlassIcon, ArrowPathIcon } from '@heroicons/react/24/outline';
import { apiFetch } from '../utils/apiFetch';

// ── Types ────────────────────────────────────────────────────────────────────

interface KGNode {
  id: string;
  label: string;
  type: string;
  confidence: number;
  access_count: number;
  // Physics state (mutable, not React state)
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
}

interface KGEdge {
  source: string;
  target: string;
  type: string;
  weight: number;
}

// ── Constants ────────────────────────────────────────────────────────────────

const NODE_COLORS: Record<string, string> = {
  concept: '#a855f7',
  entity:  '#3b82f6',
  fact:    '#22c55e',
  event:   '#f97316',
  default: '#94a3b8',
};

const NODE_COLORS_GLOW: Record<string, string> = {
  concept: 'rgba(168,85,247,0.35)',
  entity:  'rgba(59,130,246,0.35)',
  fact:    'rgba(34,197,94,0.35)',
  event:   'rgba(249,115,22,0.35)',
  default: 'rgba(148,163,184,0.35)',
};

const TYPE_LABELS: Record<string, string> = {
  concept: 'Concept',
  entity:  'Entity',
  fact:    'Fact',
  event:   'Event',
};

const ALL_TYPES = ['concept', 'entity', 'fact', 'event'] as const;

// Physics params
const REPULSION    = 4500;
const SPRING_K     = 0.04;
const SPRING_LEN   = 140;
const DAMPING      = 0.82;
const MIN_RADIUS   = 10;
const MAX_RADIUS   = 28;

// ── Physics helpers ───────────────────────────────────────────────────────────

function computeRadius(node: { access_count: number }, allNodes: { access_count: number }[]): number {
  const maxAcc = Math.max(1, ...allNodes.map(n => n.access_count));
  const t = node.access_count / maxAcc;
  return MIN_RADIUS + t * (MAX_RADIUS - MIN_RADIUS);
}

function scatter(nodes: KGNode[], w: number, h: number) {
  const cx = w / 2;
  const cy = h / 2;
  nodes.forEach((n, i) => {
    const angle = (i / nodes.length) * Math.PI * 2;
    const r = Math.min(w, h) * 0.3;
    n.x = cx + r * Math.cos(angle) + (Math.random() - 0.5) * 60;
    n.y = cy + r * Math.sin(angle) + (Math.random() - 0.5) * 60;
    n.vx = 0;
    n.vy = 0;
  });
}

// ── Detail Panel ──────────────────────────────────────────────────────────────

function NodeDetail({
  node,
  edges,
  allNodes,
  onClose,
  onFocus,
}: {
  node: KGNode;
  edges: KGEdge[];
  allNodes: KGNode[];
  onClose: () => void;
  onFocus: (id: string) => void;
}) {
  const connections = edges.filter(e => e.source === node.id || e.target === node.id);
  const color = NODE_COLORS[node.type] ?? NODE_COLORS.default;

  return (
    <div
      className="absolute top-4 right-4 w-64 rounded-xl border border-chat-border/40 shadow-2xl z-20"
      style={{ background: 'var(--surface-2)', backdropFilter: 'blur(12px)' }}
    >
      {/* Header */}
      <div className="flex items-start justify-between px-4 pt-3 pb-2 border-b border-chat-border/30">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-0.5">
            <span
              className="w-2.5 h-2.5 rounded-full shrink-0"
              style={{ background: color }}
            />
            <span className="text-[10px] font-semibold uppercase tracking-wider" style={{ color }}>
              {TYPE_LABELS[node.type] ?? node.type}
            </span>
          </div>
          <p className="text-sm font-semibold text-chat-text truncate">{node.label}</p>
        </div>
        <button
          onClick={onClose}
          className="text-chat-text-secondary hover:text-chat-text ml-2 text-lg leading-none"
        >
          &times;
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 gap-2 px-4 py-2.5 border-b border-chat-border/20">
        <div>
          <p className="text-[10px] text-chat-text-secondary">Confidence</p>
          <p className="text-sm font-semibold text-chat-text">{(node.confidence * 100).toFixed(0)}%</p>
        </div>
        <div>
          <p className="text-[10px] text-chat-text-secondary">Accesses</p>
          <p className="text-sm font-semibold text-chat-text">{node.access_count}</p>
        </div>
      </div>

      {/* Connections */}
      {connections.length > 0 && (
        <div className="px-4 py-2.5">
          <p className="text-[10px] font-semibold text-chat-text-secondary/60 uppercase tracking-wider mb-2">
            Connections ({connections.length})
          </p>
          <div className="space-y-1.5 max-h-36 overflow-y-auto">
            {connections.map((e, i) => {
              const peerId = e.source === node.id ? e.target : e.source;
              const peer = allNodes.find(n => n.id === peerId);
              const dir = e.source === node.id ? '→' : '←';
              return (
                <button
                  key={i}
                  onClick={() => onFocus(peerId)}
                  className="w-full flex items-center gap-2 text-left group"
                >
                  <span className="text-chat-text-secondary/40 text-xs shrink-0">{dir}</span>
                  <span
                    className="w-1.5 h-1.5 rounded-full shrink-0"
                    style={{ background: NODE_COLORS[peer?.type ?? 'default'] ?? NODE_COLORS.default }}
                  />
                  <span className="text-xs text-chat-text-secondary group-hover:text-chat-text truncate">
                    {peer?.label ?? peerId}
                  </span>
                  <span className="text-[9px] text-chat-text-secondary/40 shrink-0 ml-auto">
                    {e.type}
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Tooltip ───────────────────────────────────────────────────────────────────

function Tooltip({ x, y, node }: { x: number; y: number; node: KGNode }) {
  const color = NODE_COLORS[node.type] ?? NODE_COLORS.default;
  return (
    <div
      className="absolute pointer-events-none px-2.5 py-1.5 rounded-lg text-xs border border-chat-border/30 shadow-lg z-30"
      style={{
        left: x + 12,
        top: y - 10,
        background: 'var(--surface-2)',
        backdropFilter: 'blur(8px)',
        maxWidth: 200,
      }}
    >
      <div className="flex items-center gap-1.5 mb-0.5">
        <span className="w-2 h-2 rounded-full" style={{ background: color }} />
        <span className="font-semibold text-chat-text truncate">{node.label}</span>
      </div>
      <p className="text-[10px] text-chat-text-secondary">
        {TYPE_LABELS[node.type] ?? node.type} · {(node.confidence * 100).toFixed(0)}% confidence
      </p>
    </div>
  );
}

// ── Main Component ────────────────────────────────────────────────────────────

export function KnowledgeGraphExplorer() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Graph data
  const [rawData, setRawData] = useState<{ nodes: Omit<KGNode, 'x'|'y'|'vx'|'vy'|'radius'>[]; edges: KGEdge[]; stats: Record<string, unknown> } | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Physics nodes (mutable ref, not React state for perf)
  const nodesRef = useRef<KGNode[]>([]);
  const edgesRef = useRef<KGEdge[]>([]);

  // Interaction state
  const [selectedNode, setSelectedNode] = useState<KGNode | null>(null);
  const [hoveredNode, setHoveredNode] = useState<KGNode | null>(null);
  const [hoverPos, setHoverPos] = useState({ x: 0, y: 0 });
  const [search, setSearch] = useState('');
  const [activeTypes, setActiveTypes] = useState<Set<string>>(new Set(ALL_TYPES));

  // Transform (pan/zoom)
  const transform = useRef({ x: 0, y: 0, scale: 1 });

  // Drag state
  const dragRef = useRef<{ nodeId: string | null; panStart: { x: number; y: number } | null; isPanning: boolean }>({
    nodeId: null,
    panStart: null,
    isPanning: false,
  });

  // Animation frame
  const rafRef = useRef<number>(0);
  const simRunning = useRef(true);

  // ── Fetch data ─────────────────────────────────────────────────────────────

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await apiFetch('/api/knowledge-graph');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setRawData(data);
    } catch (e: any) {
      setError(e instanceof Error ? e.message : 'Failed to load');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // ── Initialize physics nodes when data arrives ─────────────────────────────

  useEffect(() => {
    if (!rawData) return;
    const canvas = canvasRef.current;
    const w = canvas?.width ?? 800;
    const h = canvas?.height ?? 600;

    const rawNodes = rawData.nodes.filter(n => activeTypes.has(n.type === undefined ? 'default' : n.type));

    const nodes: KGNode[] = rawNodes.map(n => ({
      ...n,
      x: w / 2 + (Math.random() - 0.5) * 200,
      y: h / 2 + (Math.random() - 0.5) * 200,
      vx: 0,
      vy: 0,
      radius: 0,
    }));

    // Assign radii
    nodes.forEach(n => { n.radius = computeRadius(n, nodes); });

    const nodeIds = new Set(nodes.map(n => n.id));
    const edges = rawData.edges.filter(e => nodeIds.has(e.source) && nodeIds.has(e.target));

    scatter(nodes, w, h);
    nodesRef.current = nodes;
    edgesRef.current = edges;
    simRunning.current = true;
  }, [rawData, activeTypes]);

  // ── Canvas resize ──────────────────────────────────────────────────────────

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const resize = () => {
      const { width, height } = container.getBoundingClientRect();
      canvas.width = width;
      canvas.height = height;
      scatter(nodesRef.current, width, height);
    };

    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(container);
    return () => ro.disconnect();
  }, []);

  // ── Physics step ───────────────────────────────────────────────────────────

  const stepPhysics = useCallback(() => {
    const nodes = nodesRef.current;
    const edges = edgesRef.current;
    if (nodes.length === 0) return;

    const nodeMap = new Map<string, KGNode>(nodes.map(n => [n.id, n]));

    // Repulsion (Barnes-Hut simplified: O(n^2) for now, fine up to ~100 nodes)
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i];
        const b = nodes[j];
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const dist2 = dx * dx + dy * dy + 0.01;
        const dist = Math.sqrt(dist2);
        const force = REPULSION / dist2;
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;
        a.vx -= fx;
        a.vy -= fy;
        b.vx += fx;
        b.vy += fy;
      }
    }

    // Spring attraction along edges
    for (const edge of edges) {
      const src = nodeMap.get(edge.source);
      const tgt = nodeMap.get(edge.target);
      if (!src || !tgt) continue;
      const dx = tgt.x - src.x;
      const dy = tgt.y - src.y;
      const dist = Math.sqrt(dx * dx + dy * dy) + 0.01;
      const displacement = dist - SPRING_LEN;
      const force = SPRING_K * displacement * (edge.weight ?? 1);
      const fx = (dx / dist) * force;
      const fy = (dy / dist) * force;
      src.vx += fx;
      src.vy += fy;
      tgt.vx -= fx;
      tgt.vy -= fy;
    }

    // Integrate + dampen + clamp to canvas bounds
    const canvas = canvasRef.current;
    const W = canvas?.width ?? 800;
    const H = canvas?.height ?? 600;
    const margin = 40;

    let totalKE = 0;
    for (const n of nodes) {
      n.vx *= DAMPING;
      n.vy *= DAMPING;
      n.x += n.vx;
      n.y += n.vy;
      // Soft boundary
      if (n.x < margin) n.vx += 0.5;
      if (n.x > W - margin) n.vx -= 0.5;
      if (n.y < margin) n.vy += 0.5;
      if (n.y > H - margin) n.vy -= 0.5;
      totalKE += n.vx * n.vx + n.vy * n.vy;
    }

    // Stop sim when settled
    if (totalKE < 0.01) simRunning.current = false;
  }, []);

  // ── Render frame ───────────────────────────────────────────────────────────

  const renderFrame = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { width: W, height: H } = canvas;
    const { x: tx, y: ty, scale } = transform.current;
    const nodes = nodesRef.current;
    const edges = edgesRef.current;
    const nodeMap = new Map<string, KGNode>(nodes.map(n => [n.id, n]));

    // Determine highlighted nodes (search)
    const searchLower = search.trim().toLowerCase();
    const highlightedIds = searchLower
      ? new Set(nodes.filter(n => n.label.toLowerCase().includes(searchLower)).map(n => n.id))
      : null;

    ctx.clearRect(0, 0, W, H);

    ctx.save();
    ctx.translate(tx, ty);
    ctx.scale(scale, scale);

    // ── Draw edges ──────────────────────────────────────────────────────────
    for (const edge of edges) {
      const src = nodeMap.get(edge.source);
      const tgt = nodeMap.get(edge.target);
      if (!src || !tgt) continue;

      const dimmed = highlightedIds
        ? !highlightedIds.has(src.id) && !highlightedIds.has(tgt.id)
        : false;

      ctx.beginPath();
      ctx.moveTo(src.x, src.y);
      ctx.lineTo(tgt.x, tgt.y);
      ctx.strokeStyle = `rgba(148,163,184,${dimmed ? 0.05 : (edge.weight ?? 1) * 0.3})`;
      ctx.lineWidth = 1 / scale;
      ctx.stroke();

      // Edge label (only if zoomed in enough)
      if (scale > 0.9 && edge.type && !dimmed) {
        const mx = (src.x + tgt.x) / 2;
        const my = (src.y + tgt.y) / 2;
        ctx.save();
        ctx.font = `${10 / scale}px system-ui, sans-serif`;
        ctx.fillStyle = 'rgba(148,163,184,0.45)';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(edge.type, mx, my);
        ctx.restore();
      }
    }

    // ── Draw nodes ──────────────────────────────────────────────────────────
    for (const node of nodes) {
      const color = NODE_COLORS[node.type] ?? NODE_COLORS.default;
      const glow  = NODE_COLORS_GLOW[node.type] ?? NODE_COLORS_GLOW.default;
      const isSelected  = selectedNode?.id === node.id;
      const isHovered   = hoveredNode?.id === node.id;
      const isHighlight = highlightedIds ? highlightedIds.has(node.id) : true;
      const dimmed      = !isHighlight && !isSelected;

      // Glow for selected / highlighted
      if (isSelected || isHovered || (highlightedIds && isHighlight)) {
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius + 8 / scale, 0, Math.PI * 2);
        ctx.fillStyle = glow;
        ctx.fill();
      }

      // Circle
      ctx.beginPath();
      ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
      ctx.fillStyle = dimmed
        ? 'rgba(100,116,139,0.2)'
        : color;
      ctx.globalAlpha = dimmed ? 0.3 : 1;
      ctx.fill();

      if (isSelected) {
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2 / scale;
        ctx.stroke();
      }
      ctx.globalAlpha = 1;

      // Label
      if (!dimmed || isSelected) {
        ctx.font = `bold ${Math.max(9, 11 / scale)}px system-ui, sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ctx.fillStyle = dimmed ? 'rgba(148,163,184,0.3)' : 'rgba(226,232,240,0.9)';
        ctx.fillText(node.label, node.x, node.y + node.radius + 4 / scale);
      }
    }

    ctx.restore();
  }, [search, selectedNode, hoveredNode]);

  // ── Animation loop ──────────────────────────────────────────────────────────

  useEffect(() => {
    let frameId: number;
    const loop = () => {
      if (simRunning.current) stepPhysics();
      renderFrame();
      frameId = requestAnimationFrame(loop);
    };
    frameId = requestAnimationFrame(loop);
    rafRef.current = frameId;
    return () => cancelAnimationFrame(frameId);
  }, [stepPhysics, renderFrame]);

  // ── Hit test ───────────────────────────────────────────────────────────────

  const hitTest = useCallback((clientX: number, clientY: number): KGNode | null => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const { x: tx, y: ty, scale } = transform.current;
    const wx = (clientX - rect.left - tx) / scale;
    const wy = (clientY - rect.top  - ty) / scale;
    let closest: KGNode | null = null;
    let closestDist = Infinity;
    for (const node of nodesRef.current) {
      const dx = wx - node.x;
      const dy = wy - node.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < node.radius + 4 && dist < closestDist) {
        closest = node;
        closestDist = dist;
      }
    }
    return closest;
  }, []);

  // ── Mouse events ────────────────────────────────────────────────────────────

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const hit = hitTest(e.clientX, e.clientY);
    setHoveredNode(hit);
    if (hit) {
      const canvas = canvasRef.current!;
      const rect = canvas.getBoundingClientRect();
      setHoverPos({ x: e.clientX - rect.left, y: e.clientY - rect.top });
    }

    if (dragRef.current.nodeId) {
      // Dragging a node
      const { x: tx, y: ty, scale } = transform.current;
      const canvas = canvasRef.current!;
      const rect = canvas.getBoundingClientRect();
      const wx = (e.clientX - rect.left - tx) / scale;
      const wy = (e.clientY - rect.top  - ty) / scale;
      const node = nodesRef.current.find(n => n.id === dragRef.current.nodeId);
      if (node) {
        node.x = wx;
        node.y = wy;
        node.vx = 0;
        node.vy = 0;
        simRunning.current = true;
      }
    } else if (dragRef.current.isPanning && dragRef.current.panStart) {
      const dx = e.clientX - dragRef.current.panStart.x;
      const dy = e.clientY - dragRef.current.panStart.y;
      transform.current.x += dx;
      transform.current.y += dy;
      dragRef.current.panStart = { x: e.clientX, y: e.clientY };
    }
  }, [hitTest]);

  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const hit = hitTest(e.clientX, e.clientY);
    if (hit) {
      dragRef.current.nodeId = hit.id;
      dragRef.current.isPanning = false;
    } else {
      dragRef.current.isPanning = true;
      dragRef.current.panStart = { x: e.clientX, y: e.clientY };
    }
  }, [hitTest]);

  const handleMouseUp = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const wasNode = dragRef.current.nodeId;
    dragRef.current.nodeId = null;
    dragRef.current.isPanning = false;
    dragRef.current.panStart = null;

    // Click to select (only if didn't drag much)
    if (wasNode) {
      const hit = hitTest(e.clientX, e.clientY);
      if (hit) {
        setSelectedNode(prev => prev?.id === hit.id ? null : hit);
      }
    }
  }, [hitTest]);

  const handleWheel = useCallback((e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    const newScale = Math.max(0.2, Math.min(4, transform.current.scale * delta));
    // Zoom toward cursor
    transform.current.x = mx - (mx - transform.current.x) * (newScale / transform.current.scale);
    transform.current.y = my - (my - transform.current.y) * (newScale / transform.current.scale);
    transform.current.scale = newScale;
  }, []);

  const handleMouseLeave = useCallback(() => {
    setHoveredNode(null);
    dragRef.current.nodeId = null;
    dragRef.current.isPanning = false;
  }, []);

  // ── Focus node (from detail panel click) ───────────────────────────────────

  const focusNode = useCallback((id: string) => {
    const node = nodesRef.current.find(n => n.id === id);
    if (!node) return;
    setSelectedNode(node);
    // Pan canvas to center on node
    const canvas = canvasRef.current;
    if (!canvas) return;
    const { width: W, height: H } = canvas;
    const { scale } = transform.current;
    transform.current.x = W / 2 - node.x * scale;
    transform.current.y = H / 2 - node.y * scale;
  }, []);

  // ── Type filter toggle ──────────────────────────────────────────────────────

  const toggleType = useCallback((type: string) => {
    setActiveTypes(prev => {
      const next = new Set(prev);
      if (next.has(type)) {
        if (next.size > 1) next.delete(type); // keep at least one
      } else {
        next.add(type);
      }
      return next;
    });
    simRunning.current = true;
  }, []);

  // ── Stats ───────────────────────────────────────────────────────────────────

  const stats = rawData?.stats ?? {};
  const nodeCount = nodesRef.current.length;
  const edgeCount = edgesRef.current.length;
  const clusterCount = (stats.clusters as number | undefined) ?? '—';

  // ── Empty / loading / error states ─────────────────────────────────────────

  const isEmpty = !loading && !error && rawData && rawData.nodes.length === 0;

  return (
    <div className="h-full flex flex-col bg-chat-bg">
      {/* Stats + controls bar */}
      <div className="flex flex-wrap items-center gap-3 px-4 py-2.5 border-b border-chat-border/30 shrink-0">
        {/* Stats pills */}
        <div className="flex items-center gap-3 text-[11px] text-chat-text-secondary">
          <span><span className="font-semibold text-chat-text">{nodeCount}</span> nodes</span>
          <span className="text-chat-border/50">·</span>
          <span><span className="font-semibold text-chat-text">{edgeCount}</span> edges</span>
          <span className="text-chat-border/50">·</span>
          <span><span className="font-semibold text-chat-text">{clusterCount}</span> clusters</span>
        </div>

        {/* Type filters */}
        <div className="flex items-center gap-1.5 ml-auto">
          {ALL_TYPES.map(type => {
            const active = activeTypes.has(type);
            const color = NODE_COLORS[type];
            return (
              <button
                key={type}
                onClick={() => toggleType(type)}
                className="flex items-center gap-1 px-2 py-1 rounded-full text-[11px] font-medium transition-all"
                style={{
                  background: active ? `${color}22` : 'transparent',
                  border: `1px solid ${active ? color : 'rgba(148,163,184,0.2)'}`,
                  color: active ? color : 'rgba(148,163,184,0.5)',
                }}
              >
                <span
                  className="w-1.5 h-1.5 rounded-full"
                  style={{ background: active ? color : 'rgba(148,163,184,0.3)' }}
                />
                {TYPE_LABELS[type]}
              </button>
            );
          })}
        </div>

        {/* Search */}
        <div className="relative">
          <MagnifyingGlassIcon className="absolute left-2 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-chat-text-secondary/50" />
          <input
            type="text"
            placeholder="Find node..."
            value={search}
            onChange={e => setSearch(e.target.value)}
            className="pl-7 pr-3 py-1 rounded-lg text-xs bg-chat-border/10 border border-chat-border/20 text-chat-text placeholder-chat-text-secondary/40 outline-none focus:border-chat-accent/50 w-32"
          />
        </div>

        {/* Refresh */}
        <button
          onClick={fetchData}
          disabled={loading}
          className="p-1.5 rounded-lg text-chat-text-secondary hover:text-chat-text hover:bg-chat-border/20 disabled:opacity-40 transition-colors"
          title="Refresh"
        >
          <ArrowPathIcon className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Canvas area */}
      <div ref={containerRef} className="flex-1 relative overflow-hidden">
        <canvas
          ref={canvasRef}
          className="absolute inset-0"
          style={{ cursor: hoveredNode ? 'pointer' : dragRef.current.isPanning ? 'grabbing' : 'grab' }}
          onMouseMove={handleMouseMove}
          onMouseDown={handleMouseDown}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseLeave}
          onWheel={handleWheel}
        />

        {/* Tooltip */}
        {hoveredNode && !dragRef.current.nodeId && (
          <Tooltip x={hoverPos.x} y={hoverPos.y} node={hoveredNode} />
        )}

        {/* Node detail panel */}
        {selectedNode && (
          <NodeDetail
            node={selectedNode}
            edges={edgesRef.current}
            allNodes={nodesRef.current}
            onClose={() => setSelectedNode(null)}
            onFocus={focusNode}
          />
        )}

        {/* Loading overlay */}
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="flex flex-col items-center gap-3">
              <div className="w-8 h-8 rounded-full border-2 border-chat-accent/30 border-t-chat-accent animate-spin" />
              <p className="text-xs text-chat-text-secondary">Loading graph...</p>
            </div>
          </div>
        )}

        {/* Error state */}
        {error && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center">
              <p className="text-sm text-red-400 mb-2">Failed to load knowledge graph</p>
              <p className="text-xs text-chat-text-secondary mb-4">{error}</p>
              <button
                onClick={fetchData}
                className="px-3 py-1.5 rounded-lg text-xs bg-chat-accent text-white hover:opacity-90"
              >
                Retry
              </button>
            </div>
          </div>
        )}

        {/* Empty state */}
        {isEmpty && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center max-w-xs">
              <div className="w-16 h-16 rounded-full border-2 border-chat-border/30 flex items-center justify-center mx-auto mb-4">
                <svg className="w-8 h-8 text-chat-text-secondary/30" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <circle cx="12" cy="12" r="3" strokeWidth="2" />
                  <path d="M12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M5.6 18.4l2.1-2.1M16.3 7.7l2.1-2.1" strokeWidth="2" strokeLinecap="round" />
                </svg>
              </div>
              <p className="text-sm font-medium text-chat-text mb-1">No knowledge yet</p>
              <p className="text-xs text-chat-text-secondary">Chat with Aura to build the knowledge graph.</p>
            </div>
          </div>
        )}

        {/* Legend + controls hint */}
        {!loading && !error && !isEmpty && (
          <div className="absolute bottom-4 left-4 flex flex-col gap-1.5">
            <p className="text-[9px] text-chat-text-secondary/40 uppercase tracking-wider">Scroll to zoom · Drag to pan</p>
          </div>
        )}
      </div>
    </div>
  );
}
