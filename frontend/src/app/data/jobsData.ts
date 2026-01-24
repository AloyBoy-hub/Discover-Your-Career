export interface Job {
  id: string;
  title: string;
  company: string;
  matchScore: number;
  skillsRequired: string[];
  description: string;
  salaryRange: string;
  location: string;
}

export interface RoadmapStep {
  id: string;
  title: string;
  description: string;
  duration: string;
  type: 'skill' | 'experience' | 'certification' | 'project';
  status: 'current' | 'upcoming' | 'recommended';
}

export const mockJobs: Job[] = [
  {
    id: '1',
    title: 'Senior Frontend Developer',
    company: 'TechCorp',
    matchScore: 95,
    skillsRequired: ['React', 'TypeScript', 'Tailwind CSS', 'REST API'],
    description: 'Build modern web applications with React and TypeScript',
    salaryRange: '$120k - $160k',
    location: 'San Francisco, CA'
  },
  {
    id: '2',
    title: 'Full Stack Engineer',
    company: 'StartupXYZ',
    matchScore: 88,
    skillsRequired: ['React', 'Node.js', 'MongoDB', 'AWS'],
    description: 'Work across the stack to deliver features end-to-end',
    salaryRange: '$110k - $150k',
    location: 'Remote'
  },
  {
    id: '3',
    title: 'UI/UX Developer',
    company: 'DesignHub',
    matchScore: 85,
    skillsRequired: ['React', 'Figma', 'CSS3', 'Animations'],
    description: 'Create beautiful, responsive user interfaces',
    salaryRange: '$100k - $140k',
    location: 'New York, NY'
  },
  {
    id: '4',
    title: 'React Native Developer',
    company: 'MobileFirst',
    matchScore: 82,
    skillsRequired: ['React Native', 'JavaScript', 'iOS', 'Android'],
    description: 'Build cross-platform mobile applications',
    salaryRange: '$115k - $155k',
    location: 'Austin, TX'
  },
  {
    id: '5',
    title: 'DevOps Engineer',
    company: 'CloudScale',
    matchScore: 78,
    skillsRequired: ['Docker', 'Kubernetes', 'CI/CD', 'AWS'],
    description: 'Maintain and optimize cloud infrastructure',
    salaryRange: '$130k - $170k',
    location: 'Seattle, WA'
  },
  {
    id: '6',
    title: 'Frontend Architect',
    company: 'Enterprise Solutions',
    matchScore: 75,
    skillsRequired: ['React', 'System Design', 'Leadership', 'Performance'],
    description: 'Design and oversee frontend architecture',
    salaryRange: '$150k - $200k',
    location: 'Boston, MA'
  },
  {
    id: '7',
    title: 'JavaScript Engineer',
    company: 'WebTech',
    matchScore: 72,
    skillsRequired: ['JavaScript', 'Vue.js', 'Webpack', 'Testing'],
    description: 'Build and maintain JavaScript applications',
    salaryRange: '$105k - $145k',
    location: 'Portland, OR'
  },
  {
    id: '8',
    title: 'Technical Lead',
    company: 'InnovateLabs',
    matchScore: 70,
    skillsRequired: ['React', 'Leadership', 'Agile', 'Mentoring'],
    description: 'Lead a team of frontend developers',
    salaryRange: '$140k - $180k',
    location: 'Chicago, IL'
  },
  {
    id: '9',
    title: 'Web Performance Engineer',
    company: 'SpeedyWeb',
    matchScore: 68,
    skillsRequired: ['Performance Optimization', 'React', 'Monitoring', 'Analytics'],
    description: 'Optimize web applications for speed and efficiency',
    salaryRange: '$125k - $165k',
    location: 'Remote'
  },
  {
    id: '10',
    title: 'Frontend Platform Engineer',
    company: 'ScaleCo',
    matchScore: 65,
    skillsRequired: ['React', 'Build Tools', 'Internal Tools', 'Documentation'],
    description: 'Build internal tools and improve developer experience',
    salaryRange: '$135k - $175k',
    location: 'San Diego, CA'
  }
];

export const getRoadmapForJob = (jobId: string): RoadmapStep[] => {
  // Mock roadmap data - in a real app, this would be personalized based on CV
  const roadmaps: Record<string, RoadmapStep[]> = {
    '1': [
      {
        id: 's1',
        title: 'Master Advanced TypeScript Patterns',
        description: 'Learn generics, utility types, and advanced type manipulation to write more robust code',
        duration: '2-3 months',
        type: 'skill',
        status: 'current'
      },
      {
        id: 's2',
        title: 'Build a Complex React Application',
        description: 'Create a full-featured e-commerce platform using React, TypeScript, and modern state management',
        duration: '1-2 months',
        type: 'project',
        status: 'upcoming'
      },
      {
        id: 's3',
        title: 'Learn Performance Optimization',
        description: 'Deep dive into React performance, code splitting, lazy loading, and bundle optimization',
        duration: '3-4 weeks',
        type: 'skill',
        status: 'upcoming'
      },
      {
        id: 's4',
        title: 'Contribute to Open Source',
        description: 'Make meaningful contributions to popular React libraries to build credibility',
        duration: '2-3 months',
        type: 'experience',
        status: 'recommended'
      },
      {
        id: 's5',
        title: 'Get AWS Certification',
        description: 'Obtain AWS Certified Solutions Architect certification to understand cloud deployment',
        duration: '1-2 months',
        type: 'certification',
        status: 'recommended'
      }
    ],
    '2': [
      {
        id: 'f1',
        title: 'Learn Node.js Backend Development',
        description: 'Master Express.js, RESTful APIs, and backend architecture patterns',
        duration: '2-3 months',
        type: 'skill',
        status: 'current'
      },
      {
        id: 'f2',
        title: 'Database Management with MongoDB',
        description: 'Learn MongoDB, data modeling, indexing, and query optimization',
        duration: '1-2 months',
        type: 'skill',
        status: 'upcoming'
      },
      {
        id: 'f3',
        title: 'Build Full Stack Application',
        description: 'Create a MERN stack application from scratch with authentication and real-time features',
        duration: '2-3 months',
        type: 'project',
        status: 'upcoming'
      },
      {
        id: 'f4',
        title: 'AWS Deployment & Infrastructure',
        description: 'Learn to deploy and manage applications on AWS (EC2, S3, RDS)',
        duration: '1 month',
        type: 'skill',
        status: 'recommended'
      }
    ]
  };

  return roadmaps[jobId] || [
    {
      id: 'default1',
      title: 'Review Job Requirements',
      description: 'Carefully analyze the job posting and identify skill gaps',
      duration: '1 week',
      type: 'skill',
      status: 'current'
    },
    {
      id: 'default2',
      title: 'Build Relevant Projects',
      description: 'Create portfolio projects that demonstrate required skills',
      duration: '1-2 months',
      type: 'project',
      status: 'upcoming'
    },
    {
      id: 'default3',
      title: 'Network with Industry Professionals',
      description: 'Connect with people working in similar roles',
      duration: 'Ongoing',
      type: 'experience',
      status: 'recommended'
    }
  ];
};
