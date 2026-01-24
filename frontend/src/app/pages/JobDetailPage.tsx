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
  ChevronRight
} from 'lucide-react';
import { Button } from '@/app/components/ui/button';
import { Card } from '@/app/components/ui/card';
import { Badge } from '@/app/components/ui/badge';
import { Progress } from '@/app/components/ui/progress';
import { Separator } from '@/app/components/ui/separator';
import { mockJobs, getRoadmapForJob } from '@/app/data/jobsData';

export function JobDetailPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const navigate = useNavigate();

  const job = mockJobs.find((j) => j.id === jobId);

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

  const generateRoadmap = () => {
    return {
      currentLevel: 'Mid-Level Developer',
      targetLevel: job.title,
      estimatedTime: '6-12 months',
      phases: [
        {
          title: 'Skill Gap Analysis',
          duration: '1-2 weeks',
          tasks: [
            'Assess current technical skills against job requirements',
            'Identify 3-5 key skills to develop',
            'Create a learning priority list'
          ]
        },
        {
          title: 'Technical Skill Building',
          duration: '3-4 months',
          tasks: [
            'Complete advanced React course (e.g., Epic React)',
            'Build 2-3 portfolio projects showcasing required skills',
            'Contribute to open-source projects in relevant technologies',
            'Practice system design and architecture patterns'
          ]
        },
        {
          title: 'Professional Development',
          duration: '2-3 months',
          tasks: [
            'Attend industry meetups and conferences',
            'Network with professionals in target companies',
            'Join relevant Slack/Discord communities',
            'Start writing technical blog posts or tutorials'
          ]
        },
        {
          title: 'Application Preparation',
          duration: '2-4 weeks',
          tasks: [
            'Tailor resume to highlight relevant experience',
            'Prepare portfolio showcasing best projects',
            'Practice behavioral interview questions',
            'Review company culture and values'
          ]
        },
        {
          title: 'Interview Preparation',
          duration: '4-6 weeks',
          tasks: [
            'Practice coding challenges on LeetCode/HackerRank',
            'Complete mock interviews with peers',
            'Review system design case studies',
            'Prepare questions for interviewers'
          ]
        },
        {
          title: 'Application & Follow-up',
          duration: 'Ongoing',
          tasks: [
            'Apply to 5-10 similar positions weekly',
            'Customize cover letter for each application',
            'Follow up after 1-2 weeks',
            'Track applications in a spreadsheet'
          ]
        }
      ],
      resources: [
        {
          category: 'Online Courses',
          items: [
            'Epic React by Kent C. Dodds',
            'Frontend Masters - Advanced JavaScript',
            'Coursera - Software Architecture'
          ]
        },
        {
          category: 'Books',
          items: [
            'Clean Code by Robert Martin',
            'Designing Data-Intensive Applications',
            'The Pragmatic Programmer'
          ]
        },
        {
          category: 'Practice Platforms',
          items: [
            'LeetCode Premium',
            'System Design Primer (GitHub)',
            'Exercism for code practice'
          ]
        }
      ],
      milestones: [
        { month: 1, goal: 'Complete skill assessment and start first course' },
        { month: 3, goal: 'Finish one major portfolio project' },
        { month: 5, goal: 'Begin interview preparation and networking' },
        { month: 6, goal: 'Start active job applications' }
      ]
    };
  };

  const fullRoadmap = generateRoadmap();

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
            <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-4 flex items-center gap-4">
              <div className="w-12 h-12 bg-white/10 rounded-xl flex items-center justify-center">
                <Clock className="w-6 h-6 text-white" />
              </div>
              <div>
                <p className="text-xs text-indigo-200 uppercase font-bold tracking-wider">Timeline</p>
                <p className="text-lg font-semibold">{fullRoadmap.estimatedTime}</p>
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
            <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-4 flex items-center gap-4">
              <div className="w-12 h-12 bg-white/10 rounded-xl flex items-center justify-center">
                <MapPin className="w-6 h-6 text-white" />
              </div>
              <div>
                <p className="text-xs text-indigo-200 uppercase font-bold tracking-wider">Location</p>
                <p className="text-lg font-semibold">{job.location}</p>
              </div>
            </div>
          </div>
        </motion.div>

        <div className="flex flex-col gap-10 mt-8 pb-12">
          {/* Roadmap */}
          <div className="lg:col-span-2 space-y-8">
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
            >
              <div className="bg-white rounded-2xl shadow-xl p-8 mb-8 border border-gray-200">
                <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
                  <Target className="w-7 h-7 mr-3 text-indigo-600" />
                  Your Personalized Roadmap
                </h2>

                <div className="space-y-6">
                  {fullRoadmap.phases.map((phase, index) => (
                    <div key={index} className="relative pl-8 pb-8 border-l-4 border-indigo-200 last:border-l-0 last:pb-0">
                      {/* Phase indicator */}
                      <div className="absolute -left-3 top-0 w-6 h-6 bg-indigo-600 rounded-full flex items-center justify-center text-white text-xs font-bold">
                        {index + 1}
                      </div>

                      <div className="bg-gray-50 rounded-xl p-6">
                        <div className="flex justify-between items-start mb-3">
                          <h3 className="text-xl font-bold text-gray-900">{phase.title}</h3>
                          <span className="bg-indigo-100 text-indigo-700 px-3 py-1 rounded-full text-sm font-medium">
                            {phase.duration}
                          </span>
                        </div>

                        <ul className="space-y-2">
                          {phase.tasks.map((task, taskIndex) => (
                            <li key={taskIndex} className="flex items-start">
                              <ChevronRight className="w-5 h-5 text-indigo-600 mr-2 flex-shrink-0 mt-0.5" />
                              <span className="text-gray-700">{task}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Milestones */}
              <div className="bg-white rounded-2xl shadow-xl p-8 mb-8 border border-gray-200">
                <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
                  <Award className="w-7 h-7 mr-3 text-indigo-600" />
                  Key Milestones
                </h2>

                <div className="grid md:grid-cols-2 gap-4">
                  {fullRoadmap.milestones.map((milestone, index) => (
                    <div key={index} className="bg-gradient-to-r from-indigo-50 to-purple-50 rounded-lg p-4 border-l-4 border-indigo-600">
                      <div className="flex items-center mb-2">
                        <div className="bg-indigo-600 text-white rounded-full w-8 h-8 flex items-center justify-center font-bold text-sm mr-3">
                          M{milestone.month}
                        </div>
                        <span className="text-sm text-gray-600 font-medium">Month {milestone.month}</span>
                      </div>
                      <p className="text-gray-800">{milestone.goal}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Resources */}
              <div className="bg-white rounded-2xl shadow-xl p-8 border border-gray-200">
                <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
                  <BookOpen className="w-7 h-7 mr-3 text-indigo-600" />
                  Recommended Resources
                </h2>

                <div className="grid md:grid-cols-2 gap-6">
                  {fullRoadmap.resources.map((resource, index) => (
                    <div key={index} className="bg-gray-50 rounded-lg p-5">
                      <h3 className="font-bold text-gray-900 mb-3 text-lg">{resource.category}</h3>
                      <ul className="space-y-2">
                        {resource.items.map((item, itemIndex) => (
                          <li key={itemIndex} className="flex items-start">
                            <div className="w-1.5 h-1.5 bg-indigo-600 rounded-full mt-2 mr-2 flex-shrink-0"></div>
                            <span className="text-gray-700 text-sm">{item}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  );
}

