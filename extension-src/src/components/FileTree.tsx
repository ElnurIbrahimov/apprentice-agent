import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ChevronDown,
  ChevronRight,
  FileCode2,
  FileJson,
  FileText,
  Folder,
  FolderOpen,
  FolderPlus,
  Plus,
  Trash2,
  Pencil,
  Dot,
} from 'lucide-react';

interface TreeNode {
  children: Map<string, TreeNode>;
  isFile: boolean;
  path: string;
}

export interface FileTreeProps {
  activeFile: string;
  directories?: string[];
  files: string[];
  modifiedFiles?: Set<string>;
  onFileCreate?: (path: string) => void;
  onFileDelete?: (path: string) => void;
  onFileRename?: (oldPath: string, newPath: string) => void;
  onFileSelect: (path: string) => void;
  onFolderCreate?: (path: string) => void;
  title?: string;
}

function getFileIcon(path: string) {
  const lower = path.toLowerCase();
  if (lower.endsWith('.json')) return <FileJson size={14} />;
  if (lower.endsWith('.md') || lower.endsWith('.txt')) return <FileText size={14} />;
  return <FileCode2 size={14} />;
}

function buildTree(files: string[], directories: string[]) {
  const root: TreeNode = { children: new Map(), isFile: false, path: '' };

  const ensurePath = (path: string, isFile: boolean) => {
    const segments = path.split('/').filter(Boolean);
    let current = root;
    let currentPath = '';
    segments.forEach((segment, index) => {
      currentPath = currentPath ? `${currentPath}/${segment}` : segment;
      if (!current.children.has(segment)) {
        current.children.set(segment, {
          children: new Map(),
          isFile: isFile && index === segments.length - 1,
          path: currentPath,
        });
      }
      current = current.children.get(segment)!;
      if (isFile && index === segments.length - 1) {
        current.isFile = true;
      }
    });
  };

  directories.forEach((dir) => ensurePath(dir, false));
  files.forEach((file) => ensurePath(file, true));
  return root;
}

interface ContextMenuState {
  x: number;
  y: number;
  path: string;
  isFolder: boolean;
}

function ContextMenu({ menu, onClose, onFileCreate, onFolderCreate, onFileRename, onFileDelete }: {
  menu: ContextMenuState;
  onClose: () => void;
  onFileCreate?: (path: string) => void;
  onFolderCreate?: (path: string) => void;
  onFileRename?: (oldPath: string, newPath: string) => void;
  onFileDelete?: (path: string) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [onClose]);

  const items: Array<{ label: string; icon: React.ReactNode; action: () => void }> = [];
  const parentDir = menu.isFolder ? menu.path : menu.path.split('/').slice(0, -1).join('/');

  if (onFileCreate) items.push({ label: 'New File', icon: <Plus size={12} />, action: () => { onFileCreate(parentDir); onClose(); } });
  if (onFolderCreate) items.push({ label: 'New Folder', icon: <FolderPlus size={12} />, action: () => { onFolderCreate(parentDir); onClose(); } });
  if (!menu.isFolder && onFileRename) items.push({ label: 'Rename', icon: <Pencil size={12} />, action: () => { onFileRename(menu.path, menu.path); onClose(); } });
  if (onFileDelete) items.push({ label: 'Delete', icon: <Trash2 size={12} />, action: () => { onFileDelete(menu.path); onClose(); } });

  if (!items.length) return null;

  return (
    <div
      ref={ref}
      style={{
        position: 'fixed', left: menu.x, top: menu.y, zIndex: 9999,
        background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 'var(--r-sm)',
        boxShadow: '0 4px 12px rgba(0,0,0,0.3)', padding: '4px 0', minWidth: 140,
      }}
    >
      {items.map((item) => (
        <button
          key={item.label}
          onClick={item.action}
          style={{
            display: 'flex', alignItems: 'center', gap: 8, width: '100%',
            padding: '6px 12px', background: 'none', border: 'none',
            color: item.label === 'Delete' ? '#ef4444' : 'var(--tx)',
            fontSize: '11px', cursor: 'pointer', fontFamily: 'inherit', textAlign: 'left',
          }}
          onMouseEnter={e => { e.currentTarget.style.background = 'var(--pg)'; }}
          onMouseLeave={e => { e.currentTarget.style.background = 'none'; }}
        >
          {item.icon} {item.label}
        </button>
      ))}
    </div>
  );
}

export default function FileTree({
  activeFile,
  directories = [],
  files,
  modifiedFiles,
  onFileCreate,
  onFileDelete,
  onFileRename,
  onFileSelect,
  onFolderCreate,
  title = 'Project Files',
}: FileTreeProps) {
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set(['']));
  const [contextMenu, setContextMenu] = useState<ContextMenuState | null>(null);

  const handleContextMenu = useCallback((e: React.MouseEvent, path: string, isFolder: boolean) => {
    e.preventDefault();
    e.stopPropagation();
    setContextMenu({ x: e.clientX, y: e.clientY, path, isFolder });
  }, []);

  useEffect(() => {
    setExpandedFolders((prev) => {
      const next = new Set(prev);
      directories.forEach((dir) => {
        const parts = dir.split('/');
        let current = '';
        parts.forEach((part) => {
          current = current ? `${current}/${part}` : part;
          next.add(current);
        });
      });
      return next;
    });
  }, [directories]);

  const tree = useMemo(() => buildTree(files, directories), [directories, files]);

  const toggleFolder = (path: string) => {
    setExpandedFolders((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
  };

  const renderNode = (name: string, node: TreeNode, depth: number) => {
    const isFolder = !node.isFile;
    const isExpanded = expandedFolders.has(node.path);

    if (isFolder) {
      const children = Array.from(node.children.entries()).sort(([aName, aNode], [bName, bNode]) => {
        if (aNode.isFile !== bNode.isFile) return aNode.isFile ? 1 : -1;
        return aName.localeCompare(bName);
      });

      return (
        <div key={node.path || name}>
          <div
            onContextMenu={(e) => handleContextMenu(e, node.path, true)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 4,
              padding: '6px 8px',
              paddingLeft: 8 + depth * 14,
              color: 'var(--mu)',
              fontSize: '11px',
            }}
          >
            <button
              onClick={() => toggleFolder(node.path)}
              style={{ background: 'none', border: 'none', color: 'inherit', cursor: 'pointer', display: 'flex', padding: 0 }}
              title={isExpanded ? 'Collapse folder' : 'Expand folder'}
            >
              {isExpanded ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
            </button>
            <button
              onClick={() => toggleFolder(node.path)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                flex: 1,
                background: 'none',
                border: 'none',
                color: 'inherit',
                cursor: 'pointer',
                fontSize: 'inherit',
                fontFamily: 'inherit',
                textAlign: 'left',
              }}
              title={node.path}
            >
              {isExpanded ? <FolderOpen size={14} /> : <Folder size={14} />}
              <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{name || 'root'}</span>
            </button>
            {onFileCreate && (
              <button
                onClick={() => onFileCreate(node.path)}
                style={{ background: 'none', border: 'none', color: 'inherit', cursor: 'pointer', display: 'flex', padding: 0 }}
                title="New file"
              >
                <Plus size={12} />
              </button>
            )}
            {onFolderCreate && (
              <button
                onClick={() => onFolderCreate(node.path)}
                style={{ background: 'none', border: 'none', color: 'inherit', cursor: 'pointer', display: 'flex', padding: 0 }}
                title="New folder"
              >
                <FolderPlus size={12} />
              </button>
            )}
          </div>
          {isExpanded && children.map(([childName, childNode]) => renderNode(childName, childNode, depth + 1))}
        </div>
      );
    }

    const isActive = activeFile === node.path;
    const isModified = modifiedFiles?.has(node.path);

    return (
      <div
        key={node.path}
        onContextMenu={(e) => handleContextMenu(e, node.path, false)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          padding: '6px 8px',
          paddingLeft: 26 + depth * 14,
          background: isActive ? 'rgba(124,58,237,0.16)' : 'transparent',
          borderLeft: isActive ? '2px solid var(--p)' : '2px solid transparent',
        }}
      >
        <button
          onClick={() => onFileSelect(node.path)}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            flex: 1,
            minWidth: 0,
            background: 'none',
            border: 'none',
            color: isActive ? 'var(--tx)' : 'var(--mu)',
            cursor: 'pointer',
            fontSize: '11px',
            fontFamily: 'inherit',
            textAlign: 'left',
          }}
          title={node.path}
        >
          {getFileIcon(node.path)}
          <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{name}</span>
          {isModified && <Dot size={18} style={{ color: 'var(--pl)', marginLeft: 'auto' }} />}
        </button>
        {onFileRename && (
          <button
            onClick={() => onFileRename(node.path, node.path)}
            style={{ background: 'none', border: 'none', color: 'inherit', cursor: 'pointer', display: 'flex', padding: 0 }}
            title="Rename file"
          >
            <Pencil size={12} />
          </button>
        )}
        {onFileDelete && (
          <button
            onClick={() => onFileDelete(node.path)}
            style={{ background: 'none', border: 'none', color: 'inherit', cursor: 'pointer', display: 'flex', padding: 0 }}
            title="Delete file"
          >
            <Trash2 size={12} />
          </button>
        )}
      </div>
    );
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: 0, height: '100%' }}>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          padding: '10px 10px 8px',
          borderBottom: '1px solid var(--b1)',
          color: 'var(--tx)',
        }}
      >
        <span style={{ fontSize: '11px', fontWeight: 700, letterSpacing: '0.04em', textTransform: 'uppercase' }}>
          {title}
        </span>
        <div style={{ flex: 1 }} />
        {onFileCreate && (
          <button
            onClick={() => onFileCreate('')}
            style={{ background: 'none', border: 'none', color: 'var(--mu)', cursor: 'pointer', display: 'flex', padding: 0 }}
            title="New file"
          >
            <Plus size={13} />
          </button>
        )}
        {onFolderCreate && (
          <button
            onClick={() => onFolderCreate('')}
            style={{ background: 'none', border: 'none', color: 'var(--mu)', cursor: 'pointer', display: 'flex', padding: 0 }}
            title="New folder"
          >
            <FolderPlus size={13} />
          </button>
        )}
      </div>
      <div style={{ flex: 1, minHeight: 0, overflow: 'auto', padding: '4px 0 8px' }}>
        {Array.from(tree.children.entries())
          .sort(([aName, aNode], [bName, bNode]) => {
            if (aNode.isFile !== bNode.isFile) return aNode.isFile ? 1 : -1;
            return aName.localeCompare(bName);
          })
          .map(([name, node]) => renderNode(name, node, 0))}
        {files.length === 0 && directories.length === 0 && (
          <div style={{ padding: '14px 12px', color: 'var(--mu)', fontSize: '11px', lineHeight: 1.5 }}>
            {onFileCreate || onFolderCreate
              ? 'Create a file or ask Aura to scaffold a project.'
              : 'Waiting for streamed files to appear.'}
          </div>
        )}
      </div>
      {contextMenu && (
        <ContextMenu
          menu={contextMenu}
          onClose={() => setContextMenu(null)}
          onFileCreate={onFileCreate}
          onFolderCreate={onFolderCreate}
          onFileRename={onFileRename}
          onFileDelete={onFileDelete}
        />
      )}
    </div>
  );
}
