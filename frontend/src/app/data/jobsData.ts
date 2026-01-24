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

export interface RoadmapResource {
  title: string;
  url: string;
  type: 'video' | 'article' | 'course' | 'tool';
}

export interface RoadmapStep {
  id: string;
  title: string;
  description: string;
  duration: string;
  tasks: string[];
  resources: RoadmapResource[];
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
  // Mock roadmap data - enriched with tasks and resources
  const roadmap: RoadmapStep[] = [
    {
      id: 'step1',
      title: 'Skill Gap & Market Analysis',
      duration: '1-2 weeks',
      description: 'Analyze your current profile against market demands for this specific role.',
      tasks: [
        'Compare current CV skills with job requirements',
        'Identify top 3 technical deficiencies',
        'Research current industry salary benchmarks'
      ],
      resources: [
        { title: 'Market Trends 2024', url: '#', type: 'article' },
        { title: 'Salary Calculator', url: '#', type: 'tool' }
      ]
    },
    {
      id: 'step2',
      title: 'Core Technical Integration',
      duration: '1-2 months',
      description: 'Master the primary tech stack required for this position.',
      tasks: [
        'Complete advanced certification in core technology',
        'Build a production-ready feature using the required stack',
        'Peer review code with industry experts'
      ],
      resources: [
        { title: 'Advanced Mastery Course', url: '#', type: 'course' },
        { title: 'Tech Stack Documentation', url: '#', type: 'article' }
      ]
    },
    {
      id: 'step3',
      title: 'Professional Portfolio & Networking',
      duration: '2-3 weeks',
      description: 'Position yourself as a top candidate through visibility and networking.',
      tasks: [
        'Update LinkedIn profile with new certifications',
        'Reach out to 5 recruiters in the target industry',
        'Publish a technical blog post on a relevant topic'
      ],
      resources: [
        { title: 'Networking Strategies', url: '#', type: 'video' },
        { title: 'Portfolio Best Practices', url: '#', type: 'article' }
      ]
    }
  ];

  return roadmap;
};
