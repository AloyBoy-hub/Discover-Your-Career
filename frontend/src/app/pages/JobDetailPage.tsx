import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router';
import { motion } from 'motion/react';
import {
  ArrowLeft,
  Briefcase,
  MapPin,
  DollarSign,
  TrendingUp,
  CheckCircle2,
  Clock,
  Target,
  Award,
  BookOpen,
  Code,
  ChevronRight,
  ExternalLink,
  Circle,
  Video,
  FileText,
  Layout,
  Hammer
} from 'lucide-react';

import { Button } from '@/app/components/ui/button';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/app/components/ui/card';
import { Badge } from '@/app/components/ui/badge';
import { Progress } from '@/app/components/ui/progress';
import { Separator } from '@/app/components/ui/separator';
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/app/components/ui/accordion";

import { RoadmapStep, RoadmapResource } from '@/app/data/jobsData';
import { useFormContext } from '@/app/context/FormContext';

export function JobDetailPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const navigate = useNavigate();
  const { state } = useFormContext();

  const job = state.results?.find((j) => j.id === jobId);
  const [roadmap, setRoadmap] = useState<RoadmapStep[]>([]);
  const [isLoadingRoadmap, setIsLoadingRoadmap] = useState(false);

  useEffect(() => {
    if (!job) return;

    const fetchRoadmap = async () => {
      setIsLoadingRoadmap(true);
      try {
        // Combine skills from preferences (could also be parsed from CV if available)
        const skills = [
          ...state.preferences.techStack,
          ...state.preferences.confidentSkills
        ];

        const res = await fetch('/api/generate_roadmap', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            job_role: job.title,
            current_skills: skills
          })
        });

        if (!res.ok) throw new Error("Failed to generate roadmap");

        const data = await res.json();
        const plan = data.learning_plan; // short_term, medium_term, long_term

        // Map backend plan to Frontend RoadmapStep
        const newSteps: RoadmapStep[] = [
          {
            id: 'step1',
            title: 'Short Term: Foundation',
            duration: '1-2 Months',
            description: 'Focus on critical missing skills and immediate gaps.',
            tasks: plan.short_term?.map((t: any) => t.topic) || [],
            resources: []
          },
          {
            id: 'step2',
            title: 'Medium Term: Proficiency',
            duration: '3-6 Months',
            description: 'Deepen your knowledge and build practical experience.',
            tasks: plan.medium_term?.map((t: any) => t.topic) || [],
            resources: []
          },
          {
            id: 'step3',
            title: 'Long Term: Mastery',
            duration: '6+ Months',
            description: 'Advanced topics and specialization.',
            tasks: plan.long_term?.map((t: any) => t.topic) || [],
            resources: []
          }
        ];
        setRoadmap(newSteps);
      } catch (error) {
        console.error(error);
        // Fallback to empty if fail
        setRoadmap([]);
      } finally {
        setIsLoadingRoadmap(false);
      }
    };

    fetchRoadmap();
  }, [job, jobId, state.preferences]);

  if (!job) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl mb-2">Job not found</h2>
          <Button onClick={() => navigate('/results')}>Back to Results</Button>
        </div>
      </div>
    );
  }

  // No longer using static getRoadmapForJob directly
  const [userProgress, setUserProgress] = useState<Record<string, 'not-started' | 'in-progress' | 'completed'>>(() => {
    // Reset progress when roadmap changes
    return {};
  });

  // Effect to init progress once roadmap is loaded
  useEffect(() => {
    const initialProgress: Record<string, 'not-started' | 'in-progress' | 'completed'> = {};
    roadmap.forEach((step: RoadmapStep) => {
      initialProgress[step.id] = 'not-started';
    });
    setUserProgress(initialProgress);
  }, [roadmap]);

  const updateStageStatus = (stageId: string, status: 'not-started' | 'in-progress' | 'completed') => {
    setUserProgress(prev => ({
      ...prev,
      [stageId]: status
    }));
  };

  const calculateTotalProgress = () => {
    const totalSteps = roadmap.length;
    const completedSteps = Object.values(userProgress).filter(s => s === 'completed').length;
    const inProgressSteps = Object.values(userProgress).filter(s => s === 'in-progress').length;
    return Math.round(((completedSteps + inProgressSteps * 0.5) / totalSteps) * 100);
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircle2 className="w-6 h-6 text-green-500" />;
      case 'in-progress': return <Clock className="w-6 h-6 text-blue-500" />;
      default: return <Circle className="w-6 h-6 text-gray-300" />;
    }
  };

  const getResourceIcon = (type: string) => {
    switch (type) {
      case 'video': return '📺';
      case 'course': return '🎓';
      case 'tool': return '🛠️';
      default: return '📄';
    }
  };

  const totalProgress = calculateTotalProgress();

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <Button
          variant="ghost"
          onClick={() => navigate('/results')}
          className="mb-6 gap-2"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Results
        </Button>

        {/* Job Header Banner */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-gradient-to-r from-indigo-600 to-purple-600 rounded-3xl shadow-2xl p-10 text-white mb-2 relative overflow-hidden"
        >
          {/* Decorative background elements */}
          <div className="absolute top-0 right-0 w-64 h-64 bg-white opacity-5 -translate-y-1/2 translate-x-1/2 rounded-full blur-3xl"></div>

          <div className="relative flex justify-between items-start mb-8">
            <div>
              <div className="flex items-center gap-3 mb-4">
                <Badge className="bg-white/20 hover:bg-white/30 text-white border-transparent px-3 py-1 backdrop-blur-sm">
                  Full-Time Role
                </Badge>
              </div>
              <h1 className="text-5xl font-extrabold mb-3 tracking-tight">{job.title}</h1>
              <p className="text-2xl font-light text-indigo-100">{job.company}</p>
            </div>
            <div className="bg-white/10 backdrop-blur-md px-6 py-4 rounded-2xl border border-white/20 text-center shadow-inner">
              <p className="text-xs uppercase tracking-widest text-indigo-100 mb-1 font-bold">Match Score</p>
              <p className="text-5xl font-black">{job.matchScore}%</p>
            </div>
          </div>

          <div className="relative grid grid-cols-1 md:grid-cols-3 gap-6 mt-8 border-t border-white/10 pt-8">
            <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-4 flex items-center gap-4 col-span-1 md:col-span-2">
              <div className="flex-1">
                <div className="flex justify-between items-center mb-2">
                  <p className="text-xs text-indigo-200 uppercase font-bold tracking-wider flex items-center gap-2">
                    <Target className="w-4 h-4" />
                    Roadmap Progress
                  </p>
                  <span className="text-sm font-bold">{totalProgress}%</span>
                </div>
                <Progress value={totalProgress} className="h-2 bg-white/10" />
              </div>
            </div>
            <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-4 flex items-center gap-4">
              <div className="w-12 h-12 bg-white/10 rounded-xl flex items-center justify-center">
                <DollarSign className="w-6 h-6 text-white" />
              </div>
              <div>
                <p className="text-xs text-indigo-200 uppercase font-bold tracking-wider">Salary Range</p>
                <p className="text-lg font-semibold">{job.salaryRange}</p>
              </div>
            </div>
          </div>
        </motion.div>

        <div className="mt-8 pb-12">
          {/* Interactive Roadmap */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <Card className="border-none shadow-xl bg-white/80 backdrop-blur-sm">
              <CardHeader>
                <div className="flex items-center gap-3 mb-1">
                  <Target className="w-8 h-8 text-indigo-600" />
                  <CardTitle className="text-3xl font-bold">Interactive Career Roadmap</CardTitle>
                </div>
                <CardDescription className="text-lg ml-11">
                  Step-by-step guide to bridge your skills and land your dream role at {job.company}.
                  {isLoadingRoadmap && <span className="ml-2 text-sm text-indigo-500 animate-pulse">(Generating personalized plan...)</span>}
                </CardDescription>
              </CardHeader>
              <CardContent className="pt-6">
                <Accordion type="single" collapsible className="w-full">
                  {roadmap.map((stage: RoadmapStep, index: number) => {
                    const status = userProgress[stage.id];
                    return (
                      <motion.div
                        key={stage.id}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.3, delay: index * 0.1 }}
                      >
                        <AccordionItem value={stage.id} className="border rounded-2xl mb-4 px-6 overflow-hidden bg-white shadow-sm hover:shadow-md transition-shadow border-gray-100">
                          <AccordionTrigger className="hover:no-underline py-6">
                            <div className="flex items-center gap-6 flex-1 text-left">
                              <div className="flex-shrink-0">
                                {getStatusIcon(status)}
                              </div>
                              <div className="flex-1">
                                <div className="flex items-center gap-3 mb-1">
                                  <span className="font-bold text-xl text-gray-900">Stage {index + 1}: {stage.title}</span>
                                  <Badge variant="secondary" className="bg-indigo-50 text-indigo-700 hover:bg-indigo-100 border-none px-3 py-1">
                                    <Clock className="w-3.5 h-3.5 mr-1.5" />
                                    {stage.duration}
                                  </Badge>
                                </div>
                                <p className="text-gray-500 font-medium">{stage.description}</p>
                              </div>
                            </div>
                          </AccordionTrigger>
                          <AccordionContent>
                            <div className="space-y-8 pt-2 pb-8 ml-12">
                              <div>
                                <h4 className="text-lg font-bold mb-4 flex items-center gap-3 text-gray-800">
                                  <div className="w-8 h-8 rounded-lg bg-emerald-50 text-emerald-600 flex items-center justify-center">
                                    <BookOpen className="w-4 h-4" />
                                  </div>
                                  Core Action Items
                                </h4>
                                <ul className="space-y-3">
                                  {stage.tasks.map((task: string, idx: number) => (
                                    <li key={idx} className="text-gray-600 flex items-start gap-3">
                                      <ChevronRight className="w-5 h-5 text-emerald-500 flex-shrink-0" />
                                      <span className="font-medium">{task}</span>
                                    </li>
                                  ))}
                                </ul>
                              </div>

                              <Separator className="bg-gray-100" />

                              <div>
                                <h4 className="text-lg font-bold mb-4 flex items-center gap-3 text-gray-800">
                                  <div className="w-8 h-8 rounded-lg bg-amber-50 text-amber-600 flex items-center justify-center">
                                    <Award className="w-4 h-4" />
                                  </div>
                                  Curated Resources
                                </h4>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                  {stage.resources.map((resource: RoadmapResource, idx: number) => (
                                    <motion.div
                                      key={idx}
                                      whileHover={{ scale: 1.02 }}
                                      className="p-4 bg-gray-50/50 rounded-2xl border border-gray-100 hover:border-indigo-200 hover:bg-indigo-50/30 transition-all cursor-pointer group"
                                    >
                                      <div className="flex items-start gap-4">
                                        <div className="w-12 h-12 rounded-xl bg-white shadow-sm flex items-center justify-center text-2xl">
                                          {getResourceIcon(resource.type)}
                                        </div>
                                        <div className="flex-1">
                                          <div className="flex items-center gap-2">
                                            <a
                                              href={resource.url}
                                              target="_blank"
                                              rel="noopener noreferrer"
                                              className="font-bold text-gray-900 group-hover:text-indigo-600 transition-colors"
                                            >
                                              {resource.title}
                                            </a>
                                            <ExternalLink className="w-4 h-4 text-gray-400 group-hover:text-indigo-400" />
                                          </div>
                                          <Badge variant="secondary" className="mt-2 bg-white text-gray-600 border-gray-100 font-bold uppercase text-[10px] tracking-widest">
                                            {resource.type}
                                          </Badge>
                                        </div>
                                      </div>
                                    </motion.div>
                                  ))}
                                </div>
                              </div>

                              <div className="flex gap-3 pt-6">
                                <Button
                                  variant={status === 'completed' ? 'default' : 'outline'}
                                  onClick={() => updateStageStatus(stage.id, 'completed')}
                                  className={`rounded-xl font-bold px-6 ${status === 'completed' ? 'bg-emerald-600 hover:bg-emerald-700' : 'border-gray-200'}`}
                                >
                                  {status === 'completed' ? 'Goal Achieved!' : 'Mark as Complete'}
                                </Button>
                                <Button
                                  variant={status === 'in-progress' ? 'default' : 'outline'}
                                  onClick={() => updateStageStatus(stage.id, 'in-progress')}
                                  className={`rounded-xl font-bold px-6 ${status === 'in-progress' ? 'bg-indigo-600 hover:bg-indigo-700' : 'border-gray-200'}`}
                                >
                                  In Progress
                                </Button>
                                {status !== 'not-started' && (
                                  <Button
                                    variant="ghost"
                                    onClick={() => updateStageStatus(stage.id, 'not-started')}
                                    className="text-gray-400 hover:text-red-500 font-bold"
                                  >
                                    Reset
                                  </Button>
                                )}
                              </div>
                            </div>
                          </AccordionContent>
                        </AccordionItem>
                      </motion.div>
                    );
                  })}
                </Accordion>
              </CardContent>
            </Card>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
