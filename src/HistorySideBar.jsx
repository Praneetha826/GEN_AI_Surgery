import { useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { useToast } from '@/components/ui/use-toast';
import { X, Film, Image, Trash2 } from 'lucide-react';

const HistorySidebar = ({ onClose }) => {
  const [history, setHistory] = useState([]);
  const [selectedEntry, setSelectedEntry] = useState(null);
  const { toast } = useToast() || { toast: (args) => console.log('Toast:', args) };

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (!token) {
      toast({ variant: 'destructive', title: 'Error', description: 'Please log in to view history.' });
      return;
    }

    fetch('http://localhost:5000/auth/history', {
      headers: {
        'Authorization': `Bearer ${token}`,
        'Origin': 'http://localhost:3000'
      },
      credentials: 'include'
    })
      .then(res => res.json())
      .then(data => {
        if (data.history) {
          setHistory(data.history);
        } else {
          toast({ variant: 'destructive', title: 'Error', description: data.error || 'Failed to load history' });
        }
      })
      .catch(err => {
        console.error('History fetch error:', err);
        toast({ variant: 'destructive', title: 'Error', description: 'Error loading history' });
      });
  }, [toast]);

  const handleDelete = async (historyId, e) => {
    e.stopPropagation();
    const token = localStorage.getItem('token');
    try {
      const res = await fetch(`http://localhost:5000/auth/history/${historyId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Origin': 'http://localhost:3000'
        },
        credentials: 'include'
      });
      const data = await res.json();
      if (res.ok) {
        setHistory(history.filter(entry => entry._id !== historyId));
        toast({ title: 'Success', description: 'History entry deleted.' });
        if (selectedEntry?._id === historyId) {
          setSelectedEntry(null);
        }
      } else {
        toast({ variant: 'destructive', title: 'Error', description: data.error || 'Delete failed' });
      }
    } catch (err) {
      toast({ variant: 'destructive', title: 'Error', description: `Error deleting history: ${err.message}` });
    }
  };

  return (
    <>
      {/* Overlay - clicking this closes the sidebar */}
      <div 
        className="fixed inset-0 bg-black/50 z-40"
        onClick={onClose}
      />
      
      {/* Sidebar */}
      <div className="fixed top-20 left-0 h-[calc(100vh-5rem)] w-96 bg-white shadow-2xl z-50 flex flex-col">
        <div className="flex justify-between items-center px-4 py-3 border-b border-gray-200 bg-medical/5">
          <h2 className="text-lg font-semibold text-medical-text">Analysis History</h2>
          <Button variant="ghost" size="sm" onClick={onClose} className="hover:bg-medical/10">
            <X className="h-5 w-5" />
          </Button>
        </div>
        
        <div className="flex-1 overflow-y-auto p-3">
          {history.length === 0 ? (
            <p className="text-medical-text-light text-center mt-8">No history available.</p>
          ) : (
            <div className="space-y-3">
              {history.map((entry) => (
                <Card 
                  key={entry._id} 
                  className="cursor-pointer hover:shadow-lg transition-all border border-medical-light/30 hover:border-medical"
                  onClick={() => setSelectedEntry(entry)}
                >
                  <CardContent className="p-3">
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center flex-1 min-w-0">
                        {entry.media_type === 'video' ? (
                          <Film className="h-5 w-5 text-medical mr-2 flex-shrink-0" />
                        ) : (
                          <Image className="h-5 w-5 text-medical mr-2 flex-shrink-0" />
                        )}
                        <p className="font-semibold text-sm text-medical-text truncate">
                          {entry.model ? entry.model.replace('_', ' ').toUpperCase() : 'Combine Analysis'}
                        </p>
                      </div>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={(e) => handleDelete(entry._id, e)}
                        className="text-red-500 hover:text-red-700 hover:bg-red-50 h-7 w-7 p-0 flex-shrink-0 ml-2"
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>

                    {/* Media preview - Show both input and output for segmentation */}
                    {entry.media_type === 'image' && entry.model === 'instrument_segmentation' ? (
                      <div className="grid grid-cols-2 gap-2 mt-2 mb-2">
                        {entry.input_path && (
                          <div>
                            <p className="text-xs text-medical-text-light mb-1">Input</p>
                            <img
                              src={`http://localhost:5000/${entry.input_path}`}
                              alt="Input"
                              className="w-full h-24 object-cover rounded border border-medical-light"
                            />
                          </div>
                        )}
                        {entry.output_path && (
                          <div>
                            <p className="text-xs text-medical-text-light mb-1">Output</p>
                            <img
                              src={`http://localhost:5000/${entry.output_path}`}
                              alt="Output"
                              className="w-full h-24 object-cover rounded border border-medical-light"
                            />
                          </div>
                        )}
                      </div>
                    ) : entry.media_type === 'image' && entry.input_path ? (
                      <div className="mt-2 mb-2">
                        <img
                          src={`http://localhost:5000/${entry.input_path}`}
                          alt="Preview"
                          className="w-full h-32 object-cover rounded border border-medical-light"
                        />
                      </div>
                    ) : null}

                    {entry.media_type === 'video' && entry.input_path && (
                      <div className="mt-2 mb-2 relative">
                        <video 
                          className="w-full h-32 object-cover rounded border border-medical-light bg-black"
                          preload="metadata"
                        >
                          <source src={`http://localhost:5000/${entry.input_path}#t=0.1`} type="video/mp4" />
                        </video>
                        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                          <div className="bg-black/50 rounded-full p-2">
                            <Film className="h-6 w-6 text-white" />
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Results preview */}
                    {entry.result && (
                                            // 1. Array Results (Detection, Atomic Actions)
                    Array.isArray(entry.result) ? (
                      <div className="mt-2 bg-gray-50 rounded p-2">
                      <p className="text-xs font-semibold text-medical-text mb-1">Detected Items:</p>
                      <ul className="text-xs text-medical-text-light space-y-0.5">
                        {entry.result.slice(0, 3).map((item, idx) => (
                        <li key={idx}>• {typeof item === 'string' ? item : item.name || JSON.stringify(item)}</li> 
                        ))}
                        {entry.result.length > 3 && (
                        <li className="text-medical italic">+{entry.result.length - 3} more...</li>
                        )}
                      </ul>
                      </div>
                                            // 2. Object Results (Phase/Step/Combined Analysis)
                    ) : typeof entry.result === 'object' && entry.result.phase ? (
                    <div className="mt-2 bg-gray-50 rounded p-2">
                    <p className="text-xs font-semibold text-medical-text mb-1">Phase/Step:</p>
                    <ul className="text-xs text-medical-text-light space-y-0.5">
                                <li>• Phase: {entry.result.phase.name || 'N/A'}</li>
                                <li>• Step: {entry.result.step.name || 'N/A'}</li>
                      </ul>
                      </div>
                      ) : null 
                    )}


                    <p className="text-xs text-medical-text-light mt-2">
                      {new Date(entry.timestamp).toLocaleString()}
                    </p>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Modal for detailed view */}
      {selectedEntry && (
        <>
          <div 
            className="fixed inset-0 bg-black/70 z-[60]"
            onClick={() => setSelectedEntry(null)}
          />
          <div className="fixed inset-0 z-[60] flex items-center justify-center p-4">
            <div className="bg-white rounded-lg shadow-2xl max-w-5xl w-full max-h-[90vh] overflow-y-auto">
              <div className="sticky top-0 bg-white border-b border-gray-200 px-6 py-4 flex justify-between items-start z-10">
                <div className="flex items-center">
                  {selectedEntry.media_type === 'video' ? (
                    <Film className="h-6 w-6 text-medical mr-3" />
                  ) : (
                    <Image className="h-6 w-6 text-medical mr-3" />
                  )}
                  <div>
                    <h2 className="text-2xl font-semibold text-medical-text">
                      {selectedEntry.model.replace('_', ' ').toUpperCase()}
                    </h2>
                    <p className="text-sm text-medical-text-light mt-1">
                      {new Date(selectedEntry.timestamp).toLocaleString()}
                    </p>
                  </div>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setSelectedEntry(null)}
                  className="hover:bg-gray-100"
                >
                  <X className="h-5 w-5" />
                </Button>
              </div>

              <div className="p-6">
                {selectedEntry.input_path && (
                  <div className="mb-2">
                    <p className="text-sm font-medium text-medical-text-light">
                      <span className="font-semibold">Input:</span> {selectedEntry.input_path.split('/').pop()}
                    </p>
                  </div>
                )}
                {selectedEntry.output_path && selectedEntry.output_path !== selectedEntry.input_path && (
                  <div className="mb-4">
                    <p className="text-sm font-medium text-medical-text-light">
                      <span className="font-semibold">Output:</span> {selectedEntry.output_path.split('/').pop()}
                    </p>
                  </div>
                )}

                {selectedEntry.media_type === 'image' && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                    {selectedEntry.input_path && (
                      <div>
                        <p className="text-sm font-semibold text-medical-text mb-2">Input Image</p>
                        <img
                          src={`http://localhost:5000/${selectedEntry.input_path}`}
                          alt="Input Image"
                          className="w-full h-auto rounded-lg border-2 border-medical-light shadow-md"
                        />
                      </div>
                    )}
                    {selectedEntry.output_path && selectedEntry.output_path !== selectedEntry.input_path && (
                      <div>
                        <p className="text-sm font-semibold text-medical-text mb-2">Output Image</p>
                        <img
                          src={`http://localhost:5000/${selectedEntry.output_path}`}
                          alt="Output Image"
                          className="w-full h-auto rounded-lg border-2 border-medical-light shadow-md"
                        />
                      </div>
                    )}
                  </div>
                )}

                {selectedEntry.media_type === 'video' && selectedEntry.input_path && (
                  <div className="mt-4">
                    <p className="text-sm font-semibold text-medical-text mb-2">Video</p>
                    <div className="max-w-3xl mx-auto">
                      <video controls className="w-full rounded-lg border-2 border-medical-light shadow-md">
                        <source src={`http://localhost:5000/${selectedEntry.input_path}`} type="video/mp4" />
                      </video>
                    </div>
                  </div>
                )}

                {selectedEntry.result && selectedEntry.result.length > 0 && (
                  <div className="mt-6 p-4 bg-medical/5 rounded-lg border border-medical-light">
                    <p className="text-base font-semibold text-medical-text mb-3">Results:</p>
                    <ul className="list-disc pl-5 text-sm text-medical-text-light space-y-1.5">
                      {selectedEntry.result.map((item, idx) => (
                        <li key={idx} className="leading-relaxed">{item}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {selectedEntry.result === null && (
                  <div className="mt-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
                    <p className="text-sm text-medical-text-light italic">
                      No textual results available. Visual output is displayed above.
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </>
      )}
    </>
  );
};

export default HistorySidebar;