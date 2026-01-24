import { useState, useEffect } from 'react';
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
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchRecommendations = async () => {
      try {
        const resumeText = sessionStorage.getItem('resume_text');
        const surveyAnswers = sessionStorage.getItem('surveyAnswers');
        const preferences = sessionStorage.getItem('preferences');

        let extraInfo = "";
        if (preferences) {
          try {
            const p = JSON.parse(preferences);
            extraInfo += `Preferences: Industry ${p.industry}, Location ${p.country}/${p.region}, Roles ${p.roleType.join(',')}, Tech ${p.techStack.join(',')}, Skills ${p.confidentSkills.join(',')}.\n`;
          } catch (e) { }
        }
        if (surveyAnswers) {
          extraInfo += `Survey Answers: ${surveyAnswers}`;
        }

        console.log("Fetching recommendations with extra_info:", extraInfo);

        const response = await fetch('http://localhost:8000/recommend', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            resume_text: resumeText || " ", // Ensure not empty
            top_k_retrieve: 200,
            extra_info: extraInfo,
            use_llm_rerank: true
          }),
        });

        if (!response.ok) {
          throw new Error('Failed to fetch recommendations');
        }

        const data = await response.json();
        console.log("Received data:", data);

        if (data.results && data.results.length > 0) {
          setJobs(data.results);
        } else {
          // If no results (or older backend version), fallback to mapping top10 
          // Logic: If data.results exists, use it. Else empty.
          setJobs([]);
        }
      } catch (error) {
        console.error('Error fetching jobs:', error);
        // Fallback to mock jobs for demo continuity if backend fails
        setJobs(mockJobs);
      } finally {
        setLoading(false);
      }
    };

    fetchRecommendations();
  }, []);

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
            We have provided <span className="font-semibold text-indigo-600">{jobs.length} relevant opportunities</span> based on your profile. Click on any job to see your personalised roadmap.
          </motion.p>
        </div>

        {/* Spreading Activation Network */}
        <Card className="bg-white/80 backdrop-blur-sm border-gray-200 shadow-xl">
          {viewMode === 'network' ? (
            <div className="p-8">
              <SpreadingActivationViz jobs={jobs} />
            </div>
          ) : (
            <div className="p-6">
              <div className="grid gap-4">
                {jobs.map((job: Job, index: number) => (
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
        {job.skillsRequired.map((skill: string) => (
          <Badge key={skill} variant="outline" className="text-xs">
            {skill}
          </Badge>
        ))}
      </div>
    </Card>
  );
}
