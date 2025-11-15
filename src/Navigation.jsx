// src/Navigation.jsx
import { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Menu, X, Stethoscope, LogOut } from 'lucide-react';
import { useAuth } from './App';

const Navigation = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const location = useLocation();
  const { user, logout } = useAuth();

  const navItems = [
    { name: 'Home', path: '/' },
    { name: 'Demo', path: '/demo' },
  ];

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-background/95 backdrop-blur-sm border-b border-border">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <Link to="/" className="flex items-center space-x-2">
            <Stethoscope className="h-8 w-8 text-medical" />
            <span className="text-xl font-semibold text-medical-text">GenAI Surgical AI</span>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-8">
            {navItems.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                className={`text-sm font-medium transition-colors hover:text-medical ${
                  location.pathname === item.path
                    ? 'text-medical border-b-2 border-medical'
                    : 'text-medical-text-light'
                }`}
              >
                {item.name}
              </Link>
            ))}
            {user ? (
              <>
                <span className="text-medical-text-light">Welcome, {user.name || user.email}</span>
                <Button variant="ghost" size="sm" onClick={logout}>
                  <LogOut className="h-5 w-5 mr-2" /> Logout
                </Button>
              </>
            ) : (
              <>
                <Link
                  to="/login"
                  className={`text-sm font-medium transition-colors hover:text-medical ${
                    location.pathname === '/login'
                      ? 'text-medical border-b-2 border-medical'
                      : 'text-medical-text-light'
                  }`}
                >
                  Login
                </Link>
                <Button asChild variant="default" size="sm" className="bg-medical hover:bg-medical/90">
                  <Link to="/signup">Get Started</Link>
                </Button>
              </>
            )}
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsMenuOpen(!isMenuOpen)}
            >
              {isMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
            </Button>
          </div>
        </div>

        {/* Mobile Navigation */}
        {isMenuOpen && (
          <div className="md:hidden">
            <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3 bg-background border-t border-border">
              {navItems.map((item) => (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`block px-3 py-2 rounded-md text-base font-medium transition-colors ${
                    location.pathname === item.path
                      ? 'text-medical bg-medical-light'
                      : 'text-medical-text-light hover:text-medical'
                  }`}
                  onClick={() => setIsMenuOpen(false)}
                >
                  {item.name}
                </Link>
              ))}
              {user ? (
                <>
                  <div className="px-3 py-2 text-medical-text-light">Welcome, {user.name || user.email}</div>
                  <Button variant="ghost" className="w-full justify-start" onClick={() => { logout(); setIsMenuOpen(false); }}>
                    <LogOut className="h-5 w-5 mr-2" /> Logout
                  </Button>
                </>
              ) : (
                <>
                  <Link
                    to="/login"
                    className={`block px-3 py-2 rounded-md text-base font-medium transition-colors ${
                      location.pathname === '/login'
                        ? 'text-medical bg-medical-light'
                        : 'text-medical-text-light hover:text-medical'
                    }`}
                    onClick={() => setIsMenuOpen(false)}
                  >
                    Login
                  </Link>
                  <div className="pt-2">
                    <Button asChild className="w-full bg-medical hover:bg-medical/90">
                      <Link to="/signup" onClick={() => setIsMenuOpen(false)}>Get Started</Link>
                    </Button>
                  </div>
                </>
              )}
            </div>
          </div>
        )}
      </div>
    </nav>
  );
};

export default Navigation;