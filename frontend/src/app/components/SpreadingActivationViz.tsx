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

  // MASTER CONTROLS: Change these to shift EVERYTHING (You icon, Lines, and JobCards)
  const centerX = 46; // Decrease to shift LEFT
  const centerY = 38; // Decrease to shift UP

  // Sort jobs by match score (descending)
  const sortedJobs = [...jobs].sort((a, b) => b.matchScore - a.matchScore);

  // Calculate positions for spreading activation layout
  const getPosition = (index: number, total: number) => {
    // Top match (highest score) - placed directly below center
    if (index === 0) {
      return { x: centerX - 4 , y: centerY, angle:90 }; // Anchored relative to the master center
    }

    // Remaining jobs arranged in a circle
    const remainingCount = total - 1;
    const circleIndex = index - 1;
    const radius = 38;
    const angleStep = 360 / remainingCount;
    const angle = -90 + (circleIndex * angleStep);

    const radians = (angle * Math.PI) / 180;
    return {
      x: centerX + radius * Math.cos(radians) -3, // Shifts relative to centerX
      y: centerY + radius * Math.sin(radians), // Shifts relative to centerY
    };
  };

  const handleJobClick = (jobId: string) => {
    navigate(`/job/${jobId}`);
  };

  return (
    <div className="relative w-full" style={{ height: '900px' }}>


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
          return (
            <motion.line
              key={job.id}
              x1={`${centerX+3}%`}
              y1={`${centerY+8}%`} // Lines start from the shifted center
              x2={`${pos.x+8}%`}
              y2={`${pos.y}%`}
              stroke="url(#lineGradient)"
              strokeWidth={index === 0 ? "4" : "2"}
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
              left: `${pos.x}%`, // JobCard position is based on shifted getPosition results
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
