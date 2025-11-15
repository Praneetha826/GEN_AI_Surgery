import { useLocation, Link } from "react-router-dom";
import { useEffect } from "react";
import { Button } from "@/components/ui/button";
import { AlertTriangle } from "lucide-react";

const NotFound = () => {
  const location = useLocation();

  useEffect(() => {
    console.error("404 Error: User attempted to access non-existent route:", location.pathname);
  }, [location.pathname]);

  return (
    <div className="min-h-screen pt-20 flex items-center justify-center bg-medical-light/20">
      <div className="text-center max-w-md mx-auto px-4">
        <div className="w-20 h-20 bg-medical/10 rounded-full flex items-center justify-center mx-auto mb-6">
          <AlertTriangle className="h-10 w-10 text-medical" />
        </div>
        <h1 className="text-6xl font-bold text-medical-text mb-4">404</h1>
        <h2 className="text-2xl font-semibold text-medical-text mb-4">Page Not Found</h2>
        <p className="text-medical-text-light mb-8">
          The page you're looking for doesn't exist or has been moved.
        </p>
        <Button asChild className="bg-medical hover:bg-medical/90 medical-button">
          <Link to="/">Return to Home</Link>
        </Button>
      </div>
    </div>
  );
};

export default NotFound;