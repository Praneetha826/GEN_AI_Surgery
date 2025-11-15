import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { useAuth } from './App';
import { useToast } from '@/components/ui/use-toast';

const HistoryPage = () => {
  const { user } = useAuth();
  const { toast } = useToast() || { toast: (args) => console.log('Fallback Toast:', args) };
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchHistory = async () => {
      if (!user) {
        toast({ variant: 'destructive', title: 'Error', description: 'Please log in to view history.' });
        return;
      }

      setLoading(true);
      const token = localStorage.getItem('token');

      try {
        const res = await fetch('http://localhost:5000/predict/history', {
          headers: { 'Authorization': `Bearer ${token}` },
        });

        const data = await res.json();
        if (res.ok) {
          setHistory(data.history || []);
        } else {
          toast({ variant: 'destructive', title: 'Error', description: data.error || 'Failed to fetch history.' });
        }
      } catch (err) {
        console.error('[ERROR] Fetch history error:', err);
        toast({ variant: 'destructive', title: 'Error', description: err.message });
      } finally {
        setLoading(false);
      }
    };

    fetchHistory();
  }, [user, toast]);

  return (
    <div className="min-h-screen pt-20 px-4 sm:px-6 lg:px-8">
      <h1 className="text-3xl font-bold text-medical-text mb-8">Your Analysis History</h1>

      {loading ? (
        <p className="text-medical-text">Loading history...</p>
      ) : history.length === 0 ? (
        <p className="text-medical-text-light">No history found.</p>
      ) : (
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {history.map((item, idx) => (
            <Card key={idx} className="medical-card border-0 shadow-md">
              <CardHeader>
                <CardTitle className="text-medical-text">
                  {item.model.replace('_', ' ').toUpperCase()}
                </CardTitle>
              </CardHeader>
              <CardContent>
                {item.model.includes('image') && (
                  <img
                    src={`http://localhost:5000/${item.result_path}`}
                    alt="Result"
                    className="w-full rounded-md border border-medical-light shadow-sm mb-2"
                  />
                )}
                {item.model.includes('video') && (
                  <video
                    controls
                    className="w-full rounded-md border border-medical-light shadow-sm mb-2"
                  >
                    <source src={`http://localhost:5000/${item.result_path}`} type="video/mp4" />
                  </video>
                )}
                <p className="text-sm text-medical-text-light">
                  Uploaded at: {new Date(item.created_at).toLocaleString()}
                </p>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
};

export default HistoryPage;
