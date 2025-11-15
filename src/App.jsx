// src/App.jsx
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, useLocation, useNavigate } from "react-router-dom";
import { createContext, useContext, useEffect, useState } from "react";
import Navigation from "./Navigation";
import HomePage from "./HomePage";
import LoginPage from "./LoginPage";
import SignupPage from "./SignupPage";
import DemoPage from "./DemoPage";
import NotFoundPage from "./NotFoundPage";
import ResetPassword from "./ResetPassword";
import ErrorBoundary from "./ErrorBoundary";
import { Toaster } from "sonner";
import HistorySidebar from "./HistorySideBar";

const queryClient = new QueryClient();

const AuthContext = createContext();

export const useAuth = () => useContext(AuthContext);

const AppWrapper = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [user, setUser] = useState(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  const toggleSidebar = () => {
    console.log('Toggling sidebar, current state:', isSidebarOpen);
    setIsSidebarOpen(!isSidebarOpen);
  };

  useEffect(() => {
    const params = new URLSearchParams(location.search);
    const token = params.get("token");
    if (token) {
      localStorage.setItem("token", token);
      navigate(location.pathname, { replace: true });
    }

    const storedToken = localStorage.getItem("token");
    if (storedToken) {
      fetch("http://localhost:5000/auth/me", {
        headers: {
          Authorization: `Bearer ${storedToken}`,
          Origin: "http://localhost:3000",
        },
      })
        .then((res) => res.json())
        .then((data) => {
          if (data.user) setUser(data.user);
          else localStorage.removeItem("token");
        })
        .catch((err) => {
          console.error("Auth error:", err);
          localStorage.removeItem("token");
        });
    }
  }, [location, navigate]);

  const logout = () => {
    localStorage.removeItem("token");
    setUser(null);
    navigate("/");
  };

  return (
    <AuthContext.Provider value={{ user, setUser, logout }}>
      <ErrorBoundary>
        <div className="scroll-smooth">
          <Navigation />
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/login" element={<LoginPage />} />
            <Route path="/signup" element={<SignupPage />} />
            <Route 
              path="/demo" 
              element={
                <DemoPage 
                  toggleSidebar={toggleSidebar} 
                  isSidebarOpen={isSidebarOpen} 
                />
              } 
            />
            <Route path="/reset-password/:token" element={<ResetPassword />} />
            <Route path="*" element={<NotFoundPage />} />
          </Routes>
          {isSidebarOpen && <HistorySidebar onClose={() => setIsSidebarOpen(false)} />}
        </div>
        <Toaster 
          position="bottom-right"
          expand={false}
          duration={5000}
          toastOptions={{
            style: {
              background: 'white',
              color: 'black',
              border: '1px solid #e5e7eb'
            }
          }}
        />
      </ErrorBoundary>
    </AuthContext.Provider>
  );
};

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <BrowserRouter>
        <AppWrapper />
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;