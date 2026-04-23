import { Component, type ReactNode, type ErrorInfo } from 'react';

interface Props {
  children: ReactNode;
  /** Short label identifying this boundary — shown in the fallback. */
  label?: string;
  /** Custom fallback node. Receives no props; overrides the default UI. */
  fallback?: ReactNode;
  /** Called after the user clicks "Try again". Lets the parent refetch etc. */
  onReset?: () => void;
}

interface State {
  hasError: boolean;
  error: Error | null;
  resetKey: number;
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, error: null, resetKey: 0 };

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error, resetKey: 0 };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    const prefix = this.props.label ? `[ErrorBoundary:${this.props.label}]` : '[ErrorBoundary]';
    console.error(prefix, error, info.componentStack);
  }

  handleReset = () => {
    this.setState((s) => ({ hasError: false, error: null, resetKey: s.resetKey + 1 }));
    this.props.onReset?.();
  };

  copyDetails = () => {
    const details = `${this.props.label ?? 'unknown'}: ${this.state.error?.message ?? ''}\n\n${this.state.error?.stack ?? ''}`;
    try { navigator.clipboard?.writeText(details); } catch { /* ignore */ }
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) return this.props.fallback;
      const { label } = this.props;
      return (
        <div className="flex items-center justify-center h-full p-8 text-center">
          <div
            className="max-w-md rounded-2xl p-6"
            style={{ background: 'var(--surface-1)', border: '1px solid rgba(239,68,68,0.25)' }}
          >
            <p className="text-red-400 font-medium mb-1">
              {label ? `${label} crashed` : 'Something went wrong'}
            </p>
            <p className="text-sm text-chat-text-secondary mb-4 break-words">
              {this.state.error?.message || 'An unexpected error occurred.'}
            </p>
            <div className="flex items-center justify-center gap-2">
              <button
                onClick={this.handleReset}
                className="px-4 py-2 text-sm bg-chat-accent text-white rounded-lg hover:opacity-90 transition-opacity"
              >
                Try again
              </button>
              <button
                onClick={this.copyDetails}
                className="px-3 py-2 text-xs text-chat-text-secondary hover:text-chat-text transition-colors"
                style={{ border: '1px solid var(--border-default)', borderRadius: 8 }}
                title="Copy error details"
              >
                Copy details
              </button>
            </div>
            {label && (
              <p className="text-[10px] text-chat-text-secondary opacity-50 mt-3">
                Only the {label} surface is affected. Other tabs continue to work.
              </p>
            )}
          </div>
        </div>
      );
    }
    return <div key={this.state.resetKey}>{this.props.children}</div>;
  }
}
