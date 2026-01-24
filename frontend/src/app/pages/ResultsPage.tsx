import { useState } from 'react';
import { useNavigate } from 'react-router';
import { motion } from 'motion/react';
import { ArrowLeft, List, Network } from 'lucide-react';
import { Button } from '@/app/components/ui/button';
import { Card } from '@/app/components/ui/card';
import { Badge } from '@/app/components/ui/badge';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/app/components/ui/tabs';
import { SpreadingActivationViz } from '@/app/components/SpreadingActivationViz';
import { mockJobs, Job } from '@/app/data/jobsData';

export function ResultsPage() {
  const navigate = useNavigate();
  const [viewMode, setViewMode] = useState<'network' | 'list'>('network');

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <Button
            variant="ghost"
            onClick={() => navigate('/')}
            className="gap-2"
          >
            <ArrowLeft className="w-4 h-4" />
            Back
          </Button>

          <div className="flex items-center gap-2">
            <Button
              variant={viewMode === 'network' ? 'default' : 'outline'}
              onClick={() => setViewMode('network')}
              className="gap-2"
            >
              <Network className="w-4 h-4" />
              Network View
            </Button>
            <Button
              variant={viewMode === 'list' ? 'default' : 'outline'}
              onClick={() => setViewMode('list')}
              className="gap-2"
            >
              <List className="w-4 h-4" />
              List View
            </Button>
          </div>
        </div>

        {/* Title */}
        <div className="text-center mb-8">
          <motion.h1
            className="text-4xl mb-3 text-gray-900"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            Your Job Matches
          </motion.h1>
          <motion.p
            className="text-lg text-gray-600"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
          >
            Found <span className="font-semibold text-indigo-600">{mockJobs.length} relevant opportunities</span> based on your profile
          </motion.p>
        </div>

        {/* Main Content */}
        <Card className="bg-white/80 backdrop-blur-sm border-gray-200 shadow-xl">
          {viewMode === 'network' ? (
            <div className="p-8">
              <div className="mb-6 text-center">
                <h2 className="text-xl mb-2 text-gray-900">
                  Spreading Activation Network
                </h2>
                <p className="text-sm text-gray-600">
                  Jobs are positioned based on their relevance to your profile. Click any job to see your personalized roadmap.
                </p>
              </div>
              <SpreadingActivationViz jobs={mockJobs} />
            </div>
          ) : (
            <div className="p-6">
              <div className="grid gap-4">
                {mockJobs.map((job, index) => (
                  <motion.div
                    key={job.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.05 }}
                  >
                    <JobCard job={job} onClick={() => navigate(`/job/${job.id}`)} />
                  </motion.div>
                ))}
              </div>
            </div>
          )}
        </Card>

        {/* Legend */}
        <motion.div
          className="mt-6 flex justify-center gap-6 text-sm text-gray-600"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
        >
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-indigo-500" />
            <span>High Match (85%+)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-gray-400" />
            <span>Good Match (70-84%)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-gray-300" />
            <span>Potential Match (&lt;70%)</span>
          </div>
        </motion.div>
      </div>
    </div>
  );
}

function JobCard({ job, onClick }: { job: Job; onClick: () => void }) {
  return (
    <Card
      className="p-4 hover:shadow-lg transition-shadow cursor-pointer bg-white border-gray-200"
      onClick={onClick}
    >
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1">
          <h3 className="text-lg mb-1 text-gray-900">{job.title}</h3>
          <p className="text-sm text-gray-600">{job.company}</p>
        </div>
        <Badge
          variant={job.matchScore >= 85 ? 'default' : 'secondary'}
          className="text-sm"
        >
          {job.matchScore}% Match
        </Badge>
      </div>

      <p className="text-sm text-gray-700 mb-3">{job.description}</p>

      <div className="flex items-center justify-between mb-3">
        <span className="text-sm text-gray-600">{job.location}</span>
        <span className="text-sm font-semibold text-indigo-600">{job.salaryRange}</span>
      </div>

      <div className="flex flex-wrap gap-1">
        {job.skillsRequired.map((skill) => (
          <Badge key={skill} variant="outline" className="text-xs">
            {skill}
          </Badge>
        ))}
      </div>
    </Card>
  );
}
