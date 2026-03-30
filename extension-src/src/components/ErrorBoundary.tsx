import React from 'react';
import { AlertTriangle, RotateCcw } from 'lucide-react';

interface Props {
  children: React.ReactNode;
  panelName?: string;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export default class ErrorBoundary extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error(`[ErrorBoundary${this.props.panelName ? `: ${this.props.panelName}` : ''}]`, error, info.componentStack);
  }

  private handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
          height: '100%', gap: 16, padding: 32, textAlign: 'center',
        }}>
          <div style={{
            width: 48, height: 48, borderRadius: '50%',
            background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.2)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <AlertTriangle size={22} style={{ color: '#ef4444' }} />
          </div>
          <div>
            <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--tx)', marginBottom: 4 }}>
              Something went wrong
            </div>
            <div style={{ fontSize: '11px', color: 'var(--mu)', maxWidth: 260, lineHeight: 1.5 }}>
              {this.props.panelName ? `The ${this.props.panelName} panel` : 'This panel'} encountered an error.
            </div>
            {this.state.error && (
              <div style={{
                fontSize: '10px', color: 'var(--rd)', marginTop: 8,
                maxWidth: 280, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
              }}>
                {this.state.error.message}
              </div>
            )}
          </div>
          <button
            onClick={this.handleReset}
            style={{
              display: 'flex', alignItems: 'center', gap: 6,
              background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 'var(--r-md)',
              color: 'var(--tx)', padding: '8px 16px', cursor: 'pointer',
              fontSize: '12px', fontFamily: 'inherit', fontWeight: 500,
            }}
          >
            <RotateCcw size={13} /> Try Again
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
