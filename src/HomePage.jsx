import { useState, useEffect, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Link } from 'react-router-dom';
import { 
  Brain, 
  Zap, 
  Target, 
  Shield, 
  GraduationCap, 
  Workflow, 
  BarChart3,
  Phone,
  Mail,
  MessageSquare,
  Star,
  Send,
  Loader2,
  Sparkles
} from 'lucide-react';
import { toast } from 'sonner';
import heroImage from '@/assets/hero-medical-ai.jpg';

const Home = () => {
  const [feedback, setFeedback] = useState({
    name: '',
    email: '',
    phone: '',
    message: '',
    rating: 0,
    category: 'contact'
  });
  const [loading, setLoading] = useState(false);
  const [isVisible, setIsVisible] = useState({});
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const sectionRefs = useRef({});
  const heroRef = useRef(null);

  // Mouse parallax effect
  useEffect(() => {
    const handleMouseMove = (e) => {
      if (heroRef.current) {
        const rect = heroRef.current.getBoundingClientRect();
        const x = (e.clientX - rect.left - rect.width / 2) / rect.width;
        const y = (e.clientY - rect.top - rect.height / 2) / rect.height;
        setMousePosition({ x: x * 15, y: y * 15 });
      }
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  // Intersection Observer for scroll animations
  useEffect(() => {
    const observers = {};
    
    Object.keys(sectionRefs.current).forEach(key => {
      const element = sectionRefs.current[key];
      if (element) {
        observers[key] = new IntersectionObserver(
          ([entry]) => {
            if (entry.isIntersecting) {
              setIsVisible(prev => ({ ...prev, [key]: true }));
            }
          },
          { threshold: 0.1, rootMargin: '0px 0px -100px 0px' }
        );
        observers[key].observe(element);
      }
    });

    return () => {
      Object.values(observers).forEach(observer => observer.disconnect());
    };
  }, []);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFeedback(prev => ({ ...prev, [name]: value }));
  };

  const handleRatingChange = (rating) => {
    setFeedback(prev => ({ ...prev, rating }));
  };

  const handleSubmitFeedback = async (e) => {
    e.preventDefault();
    
    if (!feedback.name.trim() || !feedback.email.trim() || !feedback.message.trim()) {
      toast.error("Missing Fields", {
        description: "Please fill in all required fields.",
        style: {
          background: 'white',
          color: 'black',
          border: '1px solid #e5e7eb'
        }
      });
      return;
    }

    setLoading(true);

    try {
      const response = await fetch('http://localhost:5000/auth/api/feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(feedback),
      });

      const data = await response.json();

      if (response.ok) {
        toast.success("Feedback Submitted Successfully!", {
          description: "Thank you for your feedback! We have received it and will get back to you soon.",
          duration: 5000,
          style: {
            background: 'white',
            color: 'black',
            border: '1px solid #e5e7eb'
          }
        });
        
        setTimeout(() => {
          setFeedback({
            name: '',
            email: '',
            phone: '',
            message: '',
            rating: 5,
            category: 'contact'
          });
        }, 100);
      } else {
        toast.error("Submission Failed", {
          description: data.error || "Please try again later.",
          style: {
            background: 'white',
            color: 'black',
            border: '1px solid #e5e7eb'
          }
        });
      }
    } catch (error) {
      console.error('Feedback submission error:', error);
      toast.error("Network Error", {
        description: "Unable to connect to server. Please check your connection and try again.",
        style: {
          background: 'white',
          color: 'black',
          border: '1px solid #e5e7eb'
        }
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen">
      {/* Custom Keyframe Animations */}
      <style>{`
        @keyframes fadeInUp {
          from {
            opacity: 0;
            transform: translateY(30px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        
        @keyframes kenBurns {
          0%, 100% {
            transform: scale(1);
          }
          50% {
            transform: scale(1.08);
          }
        }

        @keyframes float {
          0%, 100% {
            transform: translateY(0px);
          }
          50% {
            transform: translateY(-10px);
          }
        }

        @keyframes pulse {
          0%, 100% {
            opacity: 1;
          }
          50% {
            opacity: 0.8;
          }
        }

        @keyframes shimmer {
          0% {
            background-position: -1000px 0;
          }
          100% {
            background-position: 1000px 0;
          }
        }

        .animate-kenBurns {
          animation: kenBurns 20s ease-in-out infinite;
        }

        .animate-float {
          animation: float 3s ease-in-out infinite;
        }

        .animate-shimmer {
          background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
          background-size: 1000px 100%;
          animation: shimmer 3s infinite;
        }

        /* Ripple effect */
        .ripple {
          position: relative;
          overflow: hidden;
        }

        .ripple::before {
          content: '';
          position: absolute;
          top: 50%;
          left: 50%;
          width: 0;
          height: 0;
          border-radius: 50%;
          background: rgba(255, 255, 255, 0.3);
          transform: translate(-50%, -50%);
          transition: width 0.6s, height 0.6s;
        }

        .ripple:hover::before {
          width: 300px;
          height: 300px;
        }
      `}</style>

      {/* Hero Section */}
      <section 
        ref={heroRef}
        className="relative pt-24 pb-40 overflow-hidden"
      >
        <div 
          className="absolute inset-0 bg-cover bg-center bg-no-repeat animate-kenBurns transition-transform duration-300"
          style={{ 
            backgroundImage: `url(${heroImage})`,
            transform: `translate(${mousePosition.x}px, ${mousePosition.y}px)`
          }}
        >
          <div className="absolute inset-0 bg-gradient-to-r from-blue-500/50 to-blue-700/30"></div>
        </div>
        
        <div className="relative container mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <div className="max-w-4xl mx-auto">
            <h1 className="text-4xl pt-14 sm:text-5xl lg:text-6xl font-bold text-white mb-6 leading-tight animate-[fadeInUp_0.8s_ease-out]">
              Precision Surgical Insights with{' '}
              <span className="text-medical-light animate-[pulse_2s_ease-in-out_infinite]">GenAI</span>
            </h1>
            <p className="text-xl sm:text-2xl text-white/90 mb-8 leading-relaxed animate-[fadeInUp_0.8s_ease-out_0.2s_both]">
              Leveraging advanced GenAI for real-time scene understanding, tool, and organ detection in medical images.
            </p>
            <div className="animate-[fadeInUp_0.8s_ease-out_0.4s_both]">
              <Button
                asChild
                size="lg"
                className="bg-gradient-to-r from-gray-200 via-gray-300 to-gray-100 text-medical-text font-semibold hover:from-gray-300 hover:to-gray-400 shadow-md transition-all hover:scale-105 hover:shadow-xl ripple"
              >
                <Link to="/Login">Get Started</Link>
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* How Our AI Works */}
      <section className="py-20 bg-medical-light/30" ref={el => sectionRefs.current['aiWorks'] = el}>
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className={`text-center mb-16 transition-all duration-700 ${isVisible['aiWorks'] ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'}`}>
            <h2 className="text-3xl sm:text-4xl font-bold text-medical-text mb-4">
              How Our AI Works
            </h2>
            <p className="text-lg text-medical-text-light max-w-2xl mx-auto">
              Advanced machine learning algorithms trained on thousands of surgical procedures
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {[
              { 
                icon: Brain, 
                title: 'AI Model Training', 
                desc: 'Our models are trained on diverse surgical datasets, ensuring accurate recognition across various procedures and anatomical structures.', 
                delay: '0s' 
              },
              { 
                icon: Zap, 
                title: 'Realtime Processing', 
                desc: 'Process surgical footage in real-time, providing instant feedback on instrument detection and procedural analysis.', 
                delay: '0.2s' 
              },
              { 
                icon: Target, 
                title: 'Enhanced Accuracy', 
                desc: 'Achieve over 95% accuracy in tool recognition and anatomical identification, continuously improving with each analysis.', 
                delay: '0.4s' 
              }
            ].map((item, index) => (
              <Card 
                key={index}
                className={`medical-card transition-all duration-700 border-0 hover:shadow-xl hover:-translate-y-3 group relative overflow-hidden ${
                  isVisible['aiWorks'] ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'
                }`}
                style={{ transitionDelay: item.delay }}
              >
                {/* Shimmer effect on hover */}
                <div className="absolute inset-0 animate-shimmer opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
                
                <CardContent className="p-8 text-center relative">
                  <div className="w-16 h-16 bg-medical/10 rounded-full flex items-center justify-center mx-auto mb-6 animate-float group-hover:scale-110 transition-transform duration-300">
                    <item.icon className="h-8 w-8 text-medical" />
                  </div>
                  <h3 className="text-xl font-semibold text-medical-text mb-4">{item.title}</h3>
                  <p className="text-medical-text-light leading-relaxed">
                    {item.desc}
                  </p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Benefits Section */}
      <section className="py-20" ref={el => sectionRefs.current['benefits'] = el}>
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className={`text-center mb-16 transition-all duration-700 ${isVisible['benefits'] ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'}`}>
            <h2 className="text-3xl sm:text-4xl font-bold text-medical-text mb-4">
              Benefits for Surgical Practice
            </h2>
            <p className="text-lg text-medical-text-light max-w-2xl mx-auto">
              Transform your surgical workflows with AI-powered insights
            </p>
          </div>

          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <div className="space-y-8">
              {[
                { 
                  icon: Shield, 
                  title: 'Improved Patient Safety', 
                  desc: 'Real-time monitoring and alerts help prevent complications and ensure optimal patient outcomes during surgery.', 
                  color: 'medical-success', 
                  delay: '0s' 
                },
                { 
                  icon: GraduationCap, 
                  title: 'Enhanced Training & Simulation', 
                  desc: 'Provide detailed feedback and analysis for medical education and surgical skill development.', 
                  color: 'medical-accent', 
                  delay: '0.1s' 
                },
                { 
                  icon: Workflow, 
                  title: 'Streamlined Workflows', 
                  desc: 'Automate documentation and analysis, reducing administrative burden and improving efficiency.', 
                  color: 'medical', 
                  delay: '0.2s' 
                },
                { 
                  icon: BarChart3, 
                  title: 'Post-operative Analysis', 
                  desc: 'Comprehensive surgical review and performance analytics for continuous improvement.', 
                  color: 'medical-accent', 
                  delay: '0.3s' 
                }
              ].map((item, index) => (
                <div 
                  key={index}
                  className={`flex items-start space-x-4 transition-all duration-700 hover:translate-x-2 ${
                    isVisible['benefits'] ? 'opacity-100 translate-x-0' : 'opacity-0 -translate-x-10'
                  }`}
                  style={{ transitionDelay: item.delay }}
                >
                  <div className={`w-12 h-12 bg-${item.color}/10 rounded-full flex items-center justify-center flex-shrink-0 transition-transform duration-300 hover:scale-110 hover:rotate-6`}>
                    <item.icon className={`h-6 w-6 text-${item.color}`} />
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold text-medical-text mb-2">{item.title}</h3>
                    <p className="text-medical-text-light">
                      {item.desc}
                    </p>
                  </div>
                </div>
              ))}
            </div>

            <Card className={`medical-card p-8 border-0 transition-all duration-700 hover:shadow-2xl ${
              isVisible['benefits'] ? 'opacity-100 translate-x-0' : 'opacity-0 translate-x-10'
            }`}>
              <CardContent className="p-0">
                <h3 className="text-2xl font-bold text-medical-text mb-6 text-center">
                  Surgical Workflow Enhancement
                </h3>
                <div className="space-y-4">
                  {[
                    { label: 'Pre-operative Planning', color: 'medical-success', delay: '0s' },
                    { label: 'Real-time Analysis', color: 'medical-accent', delay: '0.1s' },
                    { label: 'Post-operative Review', color: 'medical', delay: '0.2s' }
                  ].map((item, index) => (
                    <div 
                      key={index}
                      className={`flex items-center justify-between p-4 bg-medical-light/50 rounded-lg transition-all duration-500 hover:bg-medical-light/70 hover:translate-x-2 ${
                        isVisible['benefits'] ? 'opacity-100 translate-x-0' : 'opacity-0 -translate-x-5'
                      }`}
                      style={{ transitionDelay: item.delay }}
                    >
                      <span className="text-medical-text font-medium">{item.label}</span>
                      <div className={`w-3 h-3 bg-${item.color} rounded-full animate-pulse`}></div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Get in Touch Section */}
      <section className="py-20 bg-medical-light/20" ref={el => sectionRefs.current['contact'] = el}>
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="max-w-2xl mx-auto">
            <div className={`text-center mb-12 transition-all duration-700 ${isVisible['contact'] ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'}`}>
              <h2 className="text-3xl sm:text-4xl font-bold text-medical-text mb-4">
                Get in Touch
              </h2>
              <p className="text-lg text-medical-text-light">
                We'd love to hear from you! Share your feedback, ask questions, or get in touch with our team.
              </p>
            </div>

            <div className={`mb-6 p-4 bg-medical/5 rounded-lg border border-medical-light/30 transition-all duration-700 ${isVisible['contact'] ? 'opacity-100 scale-100' : 'opacity-0 scale-95'}`}>
              <Label className="text-sm font-medium text-medical-text mb-2 block">
                How would you rate your experience so far? (Optional)
              </Label>
              <div className="flex justify-center space-x-1">
                {[1, 2, 3, 4, 5].map((star) => (
                  <Star
                    key={star}
                    className={`h-6 w-6 cursor-pointer transition-all duration-200 hover:scale-125 ${
                      feedback.rating >= star
                        ? 'text-yellow-400 fill-yellow-400'
                        : 'text-gray-300 hover:text-yellow-400'
                    }`}
                    onClick={() => handleRatingChange(star)}
                    fill={feedback.rating >= star ? 'currentColor' : 'none'}
                  />
                ))}
              </div>
            </div>

            <Card className={`medical-card border-0 shadow-lg transition-all duration-700 hover:shadow-2xl ${isVisible['contact'] ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'}`}>
              <CardContent className="p-8">
                <form onSubmit={handleSubmitFeedback} className="space-y-6">
                  <div className="grid md:grid-cols-2 gap-6">
                    <div className="space-y-2">
                      <Label htmlFor="name" className="text-medical-text font-medium">
                        Full Name *
                      </Label>
                      <Input 
                        id="name"
                        name="name"
                        type="text"
                        value={feedback.name}
                        onChange={handleInputChange}
                        placeholder="Enter your full name"
                        required
                        disabled={loading}
                        className="border-medical-light focus:border-medical transition-all focus:scale-[1.02]"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="phone" className="text-medical-text font-medium">
                        Phone (Optional)
                      </Label>
                      <div className="relative">
                        <Phone className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-medical-text-light transition-transform duration-300 peer-focus:scale-110" />
                        <Input 
                          id="phone"
                          name="phone"
                          type="tel"
                          value={feedback.phone}
                          onChange={handleInputChange}
                          placeholder="+1 (555) 123-4567"
                          disabled={loading}
                          className="pl-10 border-medical-light focus:border-medical transition-all focus:scale-[1.02] peer"
                        />
                      </div>
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="email" className="text-medical-text font-medium">
                      Email Address *
                    </Label>
                    <div className="relative">
                      <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-medical-text-light transition-transform duration-300 peer-focus:scale-110" />
                      <Input 
                        id="email"
                        name="email"
                        type="email"
                        value={feedback.email}
                        onChange={handleInputChange}
                        placeholder="your.email@example.com"
                        required
                        disabled={loading}
                        className="pl-10 border-medical-light focus:border-medical transition-all focus:scale-[1.02] peer"
                      />
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="message" className="text-medical-text font-medium">
                      Message *
                    </Label>
                    <div className="relative">
                      <MessageSquare className="absolute left-3 top-3 h-4 w-4 text-medical-text-light transition-transform duration-300 peer-focus:scale-110" />
                      <Textarea 
                        id="message"
                        name="message"
                        value={feedback.message}
                        onChange={handleInputChange}
                        placeholder="Tell us about your needs, feedback, or questions. We're here to help!"
                        rows={4}
                        required
                        disabled={loading}
                        className="pl-10 border-medical-light focus:border-medical resize-none transition-all focus:scale-[1.02] peer"
                      />
                    </div>
                  </div>
                  
                  <Button 
                    type="submit" 
                    disabled={loading || !feedback.name.trim() || !feedback.email.trim() || !feedback.message.trim()}
                    className="w-full bg-medical hover:bg-medical/90 disabled:bg-gray-400 disabled:cursor-not-allowed medical-button flex items-center justify-center transition-all hover:scale-105 hover:shadow-lg active:scale-95 group"
                  >
                    {loading ? (
                      <>
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                        Sending Message...
                      </>
                    ) : (
                      <>
                        <Send className="h-4 w-4 mr-2 transition-transform group-hover:translate-x-1" />
                        Send Message
                      </>
                    )}
                  </Button>
                </form>

                <p className="mt-4 pt-2 border-t border-medical-light/30 text-xs text-medical-text-light text-center">
                  * Required fields. 
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-medical-text text-white py-6">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center animate-[fadeInUp_0.8s_ease-out]">
            <p className="text-white/80 mb-4">
              © 2025 GenAI Surgical AI. All rights reserved.
            </p>
            <div className="flex justify-center space-x-6">
              <Link to="/terms" className="text-white/60 hover:text-white transition-all duration-300 hover:scale-110 relative group">
                Terms of Service
                <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-white group-hover:w-full transition-all duration-300"></span>
              </Link>
              <Link to="/privacy" className="text-white/60 hover:text-white transition-all duration-300 hover:scale-110 relative group">
                Privacy Policy
                <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-white group-hover:w-full transition-all duration-300"></span>
              </Link>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Home;