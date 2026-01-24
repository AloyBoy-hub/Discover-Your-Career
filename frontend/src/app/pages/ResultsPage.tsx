import { useState } from 'react';
import { useNavigate } from 'react-router';
import { motion } from 'motion/react';
import { ArrowLeft, DollarSign, MapPin, ChevronRight } from 'lucide-react';
import { Button } from '@/app/components/ui/button';
import { Card } from '@/app/components/ui/card';
import { Badge } from '@/app/components/ui/badge';
import { SpreadingActivationViz } from '@/app/components/SpreadingActivationViz';
import { mockJobs, Job } from '@/app/data/jobsData';

export function ResultsPage() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex items-center mb-8">
          <Button
            variant="ghost"
            onClick={() => navigate('/')}
            className="gap-2"
          >
            <ArrowLeft className="w-4 h-4" />
            Back
          </Button>
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
            We have provided <span className="font-semibold text-indigo-600">{mockJobs.length} relevant opportunities</span> based on your profile. Click on any job to see your personalised roadmap.
          </motion.p>
        </div>

        {/* Spreading Activation Network */}
        <Card className="bg-white/80 backdrop-blur-sm border-gray-200 shadow-xl mb-12">
          <div className="p-8">
            <h2 className="text-xl font-semibold mb-6 text-gray-800">Connection Mapping</h2>
            <SpreadingActivationViz jobs={mockJobs} />

            {/* Legend inside the network card */}
            <div className="mt-8 pt-8 border-t border-gray-100 flex flex-wrap justify-center gap-6 text-sm text-gray-600">
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
            </div>
          </div>
        </Card>

        {/* Detailed Job List */}
        <div className="space-y-6">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">Detailed Recommendations</h2>
          <div className="grid md:grid-cols-2 gap-6">
            {mockJobs.map((job: Job) => (
              <div
                key={job.id}
                className="bg-white rounded-2xl shadow-lg p-6 hover:shadow-xl transition-all cursor-pointer border-2 border-transparent hover:border-indigo-500 group"
                onClick={() => navigate(`/job/${job.id}`)}
              >
                <div className="flex justify-between items-start mb-4">
                  <div className="flex-1">
                    <h3 className="text-xl font-bold text-gray-900 mb-1 group-hover:text-indigo-600 transition-colors">{job.title}</h3>
                    <p className="text-gray-600 font-medium">{job.company}</p>
                  </div>
                  <div className={`px-3 py-1 rounded-full text-sm font-semibold ${job.matchScore >= 85 ? 'bg-indigo-100 text-indigo-700' : 'bg-gray-100 text-gray-700'}`}>
                    {job.matchScore}% Match
                  </div>
                </div>

                <p className="text-gray-700 mb-4 line-clamp-2">{job.description}</p>

                <div className="grid grid-cols-2 gap-3 mb-4">
                  <div className="flex items-center text-sm text-gray-600">
                    <DollarSign className="w-4 h-4 mr-2 text-gray-400" />
                    {job.salaryRange}
                  </div>
                  <div className="flex items-center text-sm text-gray-600">
                    <MapPin className="w-4 h-4 mr-2 text-gray-400" />
                    {job.location}
                  </div>
                </div>

                <div className="flex flex-wrap gap-2 mb-6">
                  {job.skillsRequired.map((skill: string, idx: number) => (
                    <span key={idx} className="bg-indigo-50 text-indigo-700 px-3 py-1 rounded-lg text-xs font-medium border border-indigo-100">
                      {skill}
                    </span>
                  ))}
                </div>

                <div className="flex items-center justify-end text-indigo-600 font-bold group-hover:translate-x-1 transition-transform">
                  View Roadmap
                  <ChevronRight className="w-5 h-5 ml-1" />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
