// src/ErrorBoundary.jsx
import { Component } from 'react';
import { Button } from '@/components/ui/button'; // Import Button

class ErrorBoundary extends Component {
  state = { hasError: false, error: null };

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center bg-medical-light/20">
          <div className="text-center">
            <h2 className="text-2xl font-bold text-medical-text">Something went wrong.</h2>
            <p className="text-medical-text-light">{this.state.error.message}</p>
            <Button onClick={() => window.location.reload()} className="mt-4">
              Reload Page
            </Button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

export default ErrorBoundary;