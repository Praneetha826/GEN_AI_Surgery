import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { useAuth } from './App';
import { useToast } from '@/components/ui/use-toast';
import { Upload, Scissors, Search, Activity, Book } from 'lucide-react';

const ENDPOINTS = {
  instrument_detection: {
    path: 'detect_image_instruments',
    fileKey: 'image',
    accept: 'image/jpeg,image/png,image/jpg,image/webp',
    inputName: 'Image',
  },
  instrument_segmentation: {
    path: 'segment_image',
    fileKey: 'image',
    accept: 'image/jpeg,image/png,image/jpg,image/webp',
    inputName: 'Image',
  },
  atomic_actions: {
    path: 'atomic_actions',
    fileKey: 'video',
    accept: 'video/mp4,video/avi,video/mov',
    inputName: 'Video',
  },
  phase_step: {
    path: 'phase_step',
    fileKey: 'video',
    accept: 'video/mp4,video/avi,video/mov',
    inputName: 'Video',
  },
  analyze_surgical_video: {
    path: 'analyze_surgical_video',
    fileKey: 'video',
    accept: 'video/mp4,video/avi,video/mov',
    inputName: 'Video',
  },
};

const DemoPage = ({ toggleSidebar, isSidebarOpen = false }) => {
  const [file, setFile] = useState(null);
  const [fileType, setFileType] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [selectedModel, setSelectedModel] = useState(null);

  const { user } = useAuth();
  const { toast } = useToast() || { toast: (args) => console.log('Fallback Toast:', args) };

  console.log('DemoPage - isSidebarOpen:', isSidebarOpen); // Debug log

  const handleFileChange = (e) => {
    const uploadedFile = e.target.files[0];
    if (!uploadedFile) {
      setFile(null);
      setFileType(null);
      return;
    }

    if (uploadedFile.type.startsWith('video/')) setFileType('video');
    else if (uploadedFile.type.startsWith('image/')) setFileType('image');
    else {
      toast({
        variant: 'destructive',
        title: 'File Error',
        description: 'Unsupported file type uploaded.',
      });
      setFile(null);
      setFileType(null);
      return;
    }

    setFile(uploadedFile);
    setResult(null);
  };

  const handlePrediction = async (modelKey) => {
    setSelectedModel(modelKey);
    const config = ENDPOINTS[modelKey];

    if (fileType !== config.fileKey) {
      toast({
        variant: 'destructive',
        title: 'Input Error',
        description: `Model "${modelKey.replace('_', ' ')}" requires a ${config.fileKey}, but a ${fileType} was uploaded.`,
      });
      return;
    }

    if (!user) {
      toast({
        variant: 'destructive',
        title: 'Error',
        description: 'Please log in to use this feature.',
      });
      return;
    }

    if (!file) {
      toast({
        variant: 'destructive',
        title: 'Error',
        description: `Please upload a ${config.inputName}.`,
      });
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append(config.fileKey, file);
    const token = localStorage.getItem('token');

    try {
      const res = await fetch(`http://localhost:5000/predict/${config.path}`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
          Origin: 'http://localhost:3000',
        },
        body: formData,
        credentials: 'include',
      });

      const data = await res.json();
      console.log('Server response:', data);

      if (res.ok) {
        setResult({ ...data, model: modelKey });
        toast({
          title: 'Success',
          description: 'Analysis complete! Results saved to history.',
        });
      } else {
        toast({
          variant: 'destructive',
          title: 'Error',
          description: data.error || 'Analysis failed',
        });
      }
    } catch (err) {
      console.error('[ERROR] Fetch error:', err);
      toast({
        variant: 'destructive',
        title: 'Error',
        description: `Error during analysis: ${err.message}`,
      });
    } finally {
      setLoading(false);
    }
  };

  const getResultSourcePath = () => {
    if (result?.model === 'instrument_segmentation' && result.output_path)
      return result.output_path;
    if (result?.model === 'instrument_detection' && result.image_path)
      return result.image_path;
    if (result?.model === 'atomic_actions' && result.input_path)
      return result.input_path;
    if (result?.model === 'phase_step' && result.video_path)
      return result.video_path.replace(/\\/g, '/');
    if (result?.model === 'analyze_surgical_video' && result.output_path)
      return result.output_path.replace(/\\/g, '/').replace(/^\/+/, '');
    return null;
  };

  const resultSourcePath = getResultSourcePath();
  const uploadedPreview = file ? URL.createObjectURL(file) : null;

  return (
    <div className="min-h-screen pt-20 pb-12 bg-medical-light/20">
      {user && !isSidebarOpen && (
        <Button
          variant="ghost"
          size="sm"
          onClick={toggleSidebar}
          className="fixed left-4 top-20 z-30 bg-medical hover:bg-medical/90 text-white shadow-lg"
        >
          <Book className="h-4 w-4 sm:h-5 sm:w-5 mr-1 sm:mr-2" />
          <span className="text-sm sm:text-base">History</span>
        </Button>
      )}

      <div
        className={`container mx-auto px-4 sm:px-6 lg:px-8 transition-all duration-300 ${isSidebarOpen ? 'ml-80 sm:ml-96' : ''
          }`}
      >
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-8">
            <h1 className="pt-8 text-3xl sm:text-4xl font-bold text-medical-text mb-4">
              Surgical AI Analysis Demo
            </h1>
            <p className="text-base sm:text-lg text-medical-text-light max-w-2xl mx-auto">
              Upload an image or video to run AI-based detection, segmentation, or action analysis.
            </p>
          </div>

          <Card className="medical-card border-0 shadow-lg">
            <CardHeader>
              <CardTitle className="flex items-center text-medical-text">
                <Upload className="h-6 w-6 mr-2 text-medical" />
                Upload Surgical Media
              </CardTitle>
            </CardHeader>

            <CardContent className="space-y-6">
              <div className="space-y-2">
                <Label htmlFor="file" className="text-medical-text font-medium">
                  Select Image or Video
                </Label>
                <Input
                  id="file"
                  type="file"
                  accept={`${ENDPOINTS.instrument_detection.accept},${ENDPOINTS.atomic_actions.accept}`}
                  onChange={handleFileChange}
                  className="border-medical-light focus:border-medical"
                />
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 sm:gap-4">
                <Button
                  onClick={() => handlePrediction('instrument_detection')}
                  disabled={loading || !file || fileType === 'video'}
                  className="bg-medical hover:bg-medical/90 medical-button flex items-center justify-center h-auto py-3 text-sm sm:text-base"
                >
                  <Search className="h-4 w-4 sm:h-5 sm:w-5 mr-2" />
                  Detect the Instruments(Image)
                </Button>
                <Button
                  onClick={() => handlePrediction('instrument_segmentation')}
                  disabled={loading || !file || fileType === 'video'}
                  className="bg-medical hover:bg-medical/90 medical-button flex items-center justify-center h-auto py-3 text-sm sm:text-base"
                >
                  <Scissors className="h-4 w-4 sm:h-5 sm:w-5 mr-2" />
                  Segment the Instruments(Image)
                </Button>
                <Button
                  onClick={() => handlePrediction('atomic_actions')}
                  disabled={loading || !file || fileType === 'image'}
                  className="bg-medical hover:bg-medical/90 medical-button flex items-center justify-center h-auto py-3 text-sm sm:text-base sm:col-span-2 lg:col-span-1"
                >
                  <Activity className="h-4 w-4 sm:h-5 sm:w-5 mr-2" />
                  Detect Minor Actions(Video)
                </Button>
                <Button
                  onClick={() => handlePrediction('phase_step')}
                  disabled={loading || !file || fileType === 'image'}
                  className="bg-medical hover:bg-medical/90 medical-button flex items-center justify-center h-auto py-3 text-sm sm:text-base"
                >
                  <Activity className="h-4 w-4 sm:h-5 sm:w-5 mr-2" />
                  Detect Major Actions(Video)
                </Button>
                <Button
                  onClick={() => handlePrediction('analyze_surgical_video')}
                  disabled={loading || !file || fileType === 'image'}
                  className="bg-medical hover:bg-medical/90 medical-button flex items-center justify-center h-auto py-3 text-sm sm:text-base sm:col-span-2 lg:col-span-1"
                >
                  <Activity className="h-4 w-4 sm:h-5 sm:w-5 mr-2" />
                  Whole Surgical Video Analysis(Video)
                </Button>

              </div>

              {loading && <p className="text-medical-text">Analyzing...</p>}

              {result && resultSourcePath && (
                <div className="mt-6">
                  <h3 className="text-lg font-semibold text-medical-text">
                    Results for {result.model.replace('_', ' ').toUpperCase()}
                  </h3>

                  {fileType === 'image' ? (
                    result.model === 'instrument_detection' ? (
                      <div className="mt-3">
                        <p className="text-sm text-medical-text-light mb-1">Input Image</p>
                        <img
                          src={`http://localhost:5000/${resultSourcePath}`}
                          alt="Input Image"
                          className="w-full max-w-full sm:max-w-2xl mx-auto mt-4 rounded-md border border-medical-light shadow-sm"
                        />
                      </div>
                    ) : (
                      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-3">
                        <div>
                          <p className="text-sm text-medical-text-light mb-1">Input Image</p>
                          <img
                            src={uploadedPreview}
                            alt="Uploaded Input"
                            className="w-full h-auto rounded-md border border-medical-light shadow-sm object-cover"
                          />
                        </div>
                        <div>
                          <p className="text-sm text-medical-text-light mb-1">Segmented Output</p>
                          <img
                            src={`http://localhost:5000/${resultSourcePath}`}
                            alt="Segmented Output"
                            className="w-full h-auto rounded-md border border-medical-light shadow-sm object-cover"
                          />
                        </div>
                      </div>
                    )
                  ) : (
                    <video
                      key={resultSourcePath}
                      controls
                      preload="auto"
                      crossOrigin="anonymous"
                      className="w-full max-w-full sm:max-w-3xl mx-auto mt-4 rounded-md border border-medical-light shadow-sm"
                    >
                      <source
                        src={`http://localhost:5000/${resultSourcePath}?v=${Date.now()}`}
                        type="video/mp4"
                      />
                      Your browser does not support the video tag.
                    </video>
                  )}

                  {result.model === 'instrument_detection' && result.instruments && (
                    <div className="mt-4">
                      <p className="font-medium text-medical-text">Detected Instruments:</p>
                      <ul className="list-disc pl-5 mt-1 text-medical-text-light text-sm">
                        {result.instruments.map((inst, idx) => (
                          <li key={idx}>{inst}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {result.model === 'atomic_actions' && result.actions && (
                    <div className="mt-4">
                      <p className="font-medium text-medical-text">Detected Actions:</p>
                      <ul className="list-disc pl-5 mt-1 text-medical-text-light text-sm">
                        {result.actions.map((action, idx) => (
                          <li key={idx}>{action}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {result && result.model === 'phase_step' && (
                    <div className="mt-6 p-4 bg-white rounded-lg shadow">
                      <h3 className="text-lg font-semibold text-medical-text">Phase-Step Results</h3>
                      <p><strong>Predicted Phase:</strong> {result.predicted_phase}</p>
                      <p><strong>Predicted Step:</strong> {result.predicted_step}</p>
                    </div>
                  )}
                  {result.model === 'analyze_surgical_video' && (
                    <div className="mt-6 p-4 bg-white rounded-lg shadow">
                      <h3 className="text-lg font-semibold text-medical-text">Combined Inference Results</h3>
                      {result.phase && <p><strong>Predicted Phase:</strong> {result.phase}</p>}
                      {result.step && <p><strong>Predicted Step:</strong> {result.step}</p>}
                      {result.actions && (
                        <>
                          <p className="font-medium text-medical-text mt-2">Detected Actions:</p>
                          <ul className="list-disc pl-5 mt-1 text-medical-text-light text-sm">
                            {result.actions.map((a, i) => <li key={i}>{a}</li>)}
                          </ul>
                        </>
                      )}
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default DemoPage;