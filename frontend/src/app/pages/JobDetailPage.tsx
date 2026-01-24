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
  Code
} from 'lucide-react';
import { Button } from '@/app/components/ui/button';
import { Card } from '@/app/components/ui/card';
import { Badge } from '@/app/components/ui/badge';
import { Progress } from '@/app/components/ui/progress';
import { Separator } from '@/app/components/ui/separator';
import { mockJobs, getRoadmapForJob, RoadmapStep } from '@/app/data/jobsData';

export function JobDetailPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const navigate = useNavigate();

  const job = mockJobs.find((j) => j.id === jobId);
  const roadmap = getRoadmapForJob(jobId || '');

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

        <div className="grid lg:grid-cols-3 gap-6">
          {/* Job Details */}
          <div className="lg:col-span-1">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
            >
              <Card className="p-6 bg-white/90 backdrop-blur-sm border-gray-200 shadow-xl sticky top-6">
                <div className="flex items-start gap-3 mb-4">
                  <div className="p-3 bg-indigo-100 rounded-lg">
                    <Briefcase className="w-6 h-6 text-indigo-600" />
                  </div>
                  <div className="flex-1">
                    <h1 className="text-2xl mb-1 text-gray-900">{job.title}</h1>
                    <p className="text-lg text-gray-600">{job.company}</p>
                  </div>
                </div>

                <div className="mb-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-gray-600">Match Score</span>
                    <span className="text-sm font-semibold text-indigo-600">
                      {job.matchScore}%
                    </span>
                  </div>
                  <Progress value={job.matchScore} className="h-2" />
                </div>

                <Separator className="my-4" />

                <div className="space-y-3">
                  <div className="flex items-center gap-3 text-gray-700">
                    <MapPin className="w-5 h-5 text-gray-400" />
                    <span>{job.location}</span>
                  </div>
                  <div className="flex items-center gap-3 text-gray-700">
                    <DollarSign className="w-5 h-5 text-gray-400" />
                    <span>{job.salaryRange}</span>
                  </div>
                </div>

                <Separator className="my-4" />

                <div>
                  <h3 className="text-sm mb-3 text-gray-900">Required Skills</h3>
                  <div className="flex flex-wrap gap-2">
                    {job.skillsRequired.map((skill) => (
                      <Badge key={skill} variant="secondary">
                        {skill}
                      </Badge>
                    ))}
                  </div>
                </div>

                <Separator className="my-4" />

                <p className="text-sm text-gray-700">{job.description}</p>

                <Button className="w-full mt-6 bg-indigo-600 hover:bg-indigo-700">
                  Apply Now
                </Button>
              </Card>
            </motion.div>
          </div>

          {/* Roadmap */}
          <div className="lg:col-span-2">
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
            >
              <Card className="p-6 bg-white/90 backdrop-blur-sm border-gray-200 shadow-xl">
                <div className="flex items-center gap-3 mb-6">
                  <TrendingUp className="w-6 h-6 text-indigo-600" />
                  <div>
                    <h2 className="text-2xl text-gray-900">Your Personalized Roadmap</h2>
                    <p className="text-sm text-gray-600">
                      Actionable steps to land this role
                    </p>
                  </div>
                </div>

                <div className="space-y-6">
                  {roadmap.map((step, index) => (
                    <RoadmapStepCard key={step.id} step={step} index={index} />
                  ))}
                </div>

                <div className="mt-8 p-4 bg-indigo-50 rounded-lg border border-indigo-200">
                  <h3 className="text-sm mb-2 text-indigo-900 flex items-center gap-2">
                    <Target className="w-4 h-4" />
                    Estimated Timeline
                  </h3>
                  <p className="text-sm text-indigo-700">
                    Following this roadmap, you could be ready to apply in approximately{' '}
                    <span className="font-semibold">4-6 months</span>
                  </p>
                </div>
              </Card>
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  );
}

function RoadmapStepCard({ step, index }: { step: RoadmapStep; index: number }) {
  const getStepIcon = (type: RoadmapStep['type']) => {
    switch (type) {
      case 'skill':
        return <Code className="w-5 h-5" />;
      case 'experience':
        return <Briefcase className="w-5 h-5" />;
      case 'certification':
        return <Award className="w-5 h-5" />;
      case 'project':
        return <BookOpen className="w-5 h-5" />;
    }
  };

  const getStatusColor = (status: RoadmapStep['status']) => {
    switch (status) {
      case 'current':
        return 'bg-indigo-500';
      case 'upcoming':
        return 'bg-blue-500';
      case 'recommended':
        return 'bg-gray-400';
    }
  };

  const getStatusBadge = (status: RoadmapStep['status']) => {
    switch (status) {
      case 'current':
        return <Badge className="bg-indigo-600">In Progress</Badge>;
      case 'upcoming':
        return <Badge variant="secondary">Next Step</Badge>;
      case 'recommended':
        return <Badge variant="outline">Recommended</Badge>;
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.1 }}
      className="relative"
    >
      {/* Timeline connector */}
      {index > 0 && (
        <div className="absolute left-6 -top-6 w-0.5 h-6 bg-gray-300" />
      )}

      <div className="flex gap-4">
        {/* Timeline dot */}
        <div className="relative shrink-0">
          <div
            className={`w-12 h-12 rounded-full ${getStatusColor(
              step.status
            )} flex items-center justify-center text-white shadow-lg`}
          >
            {getStepIcon(step.type)}
          </div>
        </div>

        {/* Content */}
        <Card className="flex-1 p-4 bg-white border-gray-200">
          <div className="flex items-start justify-between mb-2">
            <h3 className="text-lg text-gray-900">{step.title}</h3>
            {getStatusBadge(step.status)}
          </div>

          <p className="text-sm text-gray-600 mb-3">{step.description}</p>

          <div className="flex items-center gap-4 text-sm text-gray-500">
            <div className="flex items-center gap-1">
              <Clock className="w-4 h-4" />
              <span>{step.duration}</span>
            </div>
            <div className="flex items-center gap-1">
              <CheckCircle2 className="w-4 h-4" />
              <span className="capitalize">{step.type}</span>
            </div>
          </div>
        </Card>
      </div>
    </motion.div>
  );
}
