import { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Lock, Stethoscope } from 'lucide-react';
import { useToast } from "@/components/ui/use-toast";

const ResetPassword = () => {
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const { token } = useParams();
  const navigate = useNavigate();
  const toast = useToast(); // Updated: No destructuring, assuming it returns the Sonner toast function

  const handleReset = async (e) => {
    e.preventDefault();
    if (password !== confirmPassword) {
      return toast.error("Passwords do not match."); // Updated to Sonner syntax, removed extra .toast
    }
    setLoading(true);
    try {
      const res = await fetch('http://localhost:5000/auth/reset-password', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token, password }),
      });
      const data = await res.json();
      if (res.ok) {
        toast.success(data.message); // Updated to Sonner syntax, removed extra .toast
        navigate('/login');
      } else {
        toast.error(data.error); // Updated to Sonner syntax, removed extra .toast
      }
    } catch (err) {
      toast.error("An error occurred."); // Updated to Sonner syntax, removed extra .toast
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen pt-20 pb-12 bg-medical-light/20">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="max-w-md mx-auto">
          <div className="text-center mb-8">
            <div className="flex justify-center mb-4">
              <Stethoscope className="h-12 w-12 text-medical" />
            </div>
            <h1 className="text-3xl font-bold text-medical-text mb-2">Reset Password</h1>
            <p className="text-medical-text-light">Enter a new password for your account</p>
          </div>

          <Card className="medical-card border-0">
            <CardHeader className="space-y-1 pb-6">
              <CardTitle className="text-2xl font-semibold text-center text-medical-text">
                Set New Password
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <form onSubmit={handleReset} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="password" className="text-medical-text font-medium">New Password</Label>
                  <div className="relative">
                    <Lock className="absolute left-3 top-3 h-4 w-4 text-medical-text-light" />
                    <Input 
                      id="password" 
                      type="password" 
                      placeholder="Enter new password" 
                      className="pl-10 border-medical-light focus:border-medical"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      required
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="confirmPassword" className="text-medical-text font-medium">Confirm Password</Label>
                  <div className="relative">
                    <Lock className="absolute left-3 top-3 h-4 w-4 text-medical-text-light" />
                    <Input 
                      id="confirmPassword" 
                      type="password" 
                      placeholder="Confirm new password" 
                      className="pl-10 border-medical-light focus:border-medical"
                      value={confirmPassword}
                      onChange={(e) => setConfirmPassword(e.target.value)}
                      required
                    />
                  </div>
                </div>

                <Button 
                  type="submit" 
                  className="w-full bg-medical hover:bg-medical/90 medical-button"
                  disabled={loading}
                >
                  {loading ? 'Resetting...' : 'Reset Password'}
                </Button>
              </form>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default ResetPassword;