import { motion } from 'motion/react';
import { useNavigate } from 'react-router';
import { Job } from '@/app/data/jobsData';
import { Card } from '@/app/components/ui/card';
import { Badge } from '@/app/components/ui/badge';
import { MapPin, TrendingUp } from 'lucide-react';

interface SpreadingActivationVizProps {
  jobs: Job[];
}

export function SpreadingActivationViz({ jobs }: SpreadingActivationVizProps) {
  const navigate = useNavigate();

  // Sort jobs by match score (descending)
  const sortedJobs = [...jobs].sort((a, b) => b.matchScore - a.matchScore);

  // Calculate positions for spreading activation layout
  const getPosition = (index: number, total: number) => {
    // Top match (highest score) - placed directly below center
    if (index === 0) {
      return { x: 50, y: 50 }; // Center (User) placement logic handled separately in render
    }

    // For layout calculations of jobs:
    // Index 0 of sortedJobs -> Positioned below
    // Indices 1...N -> Positioned in circle

    if (index === 0) {
      return { x: 50, y: 85 }; // Best match below the center (50%)
    }

    // Remaining jobs arranged in a circle
    const remainingCount = total - 1; // Exclude top match
    const circleIndex = index - 1;
    const radius = 38; // Slightly larger to fit cards
    const angleStep = 360 / remainingCount;
    // Start from -90 deg (top) and go around, but since top is taken/avoided, let's offset
    const angle = -90 + (circleIndex * angleStep);

    // Adjust angle to avoid overlapping the bottom one (which is at 90 degrees) if needed, 
    // but equidistant circle usually works fine if n > 3.
    // Let's try uniform distribution first.

    const radians = (angle * Math.PI) / 180;
    return {
      x: 50 + radius * Math.cos(radians),
      y: 50 + radius * Math.sin(radians),
    };
  };

  const handleJobClick = (jobId: string) => {
    navigate(`/job/${jobId}`);
  };

  return (
    <div className="relative w-full" style={{ height: '900px' }}>
      {/* Center node - User Profile */}
      <motion.div
        className="absolute z-20"
        style={{
          left: '50%',
          top: '50%', // Centered vertically
          transform: 'translate(-50%, -50%)',
        }}
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        transition={{ duration: 0.5 }}
      >
        <div className="w-24 h-24 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-2xl border-4 border-white">
          <div className="text-center text-white">
            <div className="text-2xl mb-1">👤</div>
            <div className="text-xs">You</div>
          </div>
        </div>
      </motion.div>

      {/* Connection lines */}
      <svg className="absolute inset-0 w-full h-full pointer-events-none">
        <defs>
          <linearGradient id="lineGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="rgb(99, 102, 241)" stopOpacity="0.3" />
            <stop offset="100%" stopColor="rgb(168, 85, 247)" stopOpacity="0.1" />
          </linearGradient>
        </defs>
        {sortedJobs.map((job, index) => {
          const pos = getPosition(index, sortedJobs.length);
          // Center node position is fixed at 50%, 35%
          return (
            <motion.line
              key={job.id}
              x1="50%"
              y1="50%"
              x2={`${pos.x}%`}
              y2={`${pos.y}%`}
              stroke="url(#lineGradient)"
              strokeWidth={index === 0 ? "4" : "2"} // Thicker line for top match
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 1, delay: index * 0.1 }}
            />
          );
        })}
      </svg>

      {/* Job nodes */}
      {sortedJobs.map((job, index) => {
        const pos = getPosition(index, sortedJobs.length);
        const isTopMatch = index === 0;

        return (
          <motion.div
            key={job.id}
            className="absolute cursor-pointer z-10"
            style={{
              left: `${pos.x}%`,
              top: `${pos.y}%`,
              transform: 'translate(-50%, -50%)',
            }}
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.3 + index * 0.1 }}
            whileHover={{ scale: 1.05 }}
            onClick={() => handleJobClick(job.id)}
          >
            <Card className={`p-3 w-48 shadow-lg hover:shadow-xl transition-shadow border-gray-200 ${isTopMatch ? 'bg-indigo-50 border-indigo-200 ring-2 ring-indigo-500' : 'bg-white'}`}>
              {isTopMatch && (
                <div className="absolute -top-3 left-1/2 -translate-x-1/2 bg-indigo-600 text-white text-[10px] px-2 py-0.5 rounded-full font-bold uppercase tracking-wider">
                  Best Match
                </div>
              )}
              <div className="flex items-start gap-2 mb-2 pt-1">
                <div className="flex-1 min-w-0">
                  <h4 className="text-sm font-semibold text-gray-900 truncate">
                    {job.title}
                  </h4>
                  <p className="text-xs text-gray-600 truncate">{job.company}</p>
                </div>
                <Badge
                  variant={job.matchScore >= 85 ? 'default' : 'secondary'}
                  className="shrink-0 text-xs"
                >
                  {job.matchScore}%
                </Badge>
              </div>

              <div className="flex items-center gap-1 text-xs text-gray-500 mb-2">
                <MapPin className="w-3 h-3" />
                <span className="truncate">{job.location}</span>
              </div>

              <div className="flex items-center gap-1 text-xs text-indigo-600">
                <TrendingUp className="w-3 h-3" />
                <span>{job.salaryRange}</span>
              </div>

              <div className="flex flex-wrap gap-1 mt-2">
                {job.skillsRequired.slice(0, 2).map((skill) => (
                  <Badge
                    key={skill}
                    variant="outline"
                    className="text-xs px-1.5 py-0.5"
                  >
                    {skill}
                  </Badge>
                ))}
                {job.skillsRequired.length > 2 && (
                  <Badge variant="outline" className="text-xs px-1.5 py-0.5">
                    +{job.skillsRequired.length - 2}
                  </Badge>
                )}
              </div>
            </Card>
          </motion.div>
        );
      })}
    </div>
  );
}
