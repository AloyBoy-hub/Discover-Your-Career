import React, { useState, ChangeEvent, DragEvent } from 'react';
import { useNavigate } from 'react-router';
import { motion } from 'motion/react';
import { Upload, FileText, Briefcase, ChevronRight, CheckCircle2 } from 'lucide-react';
import { Button } from '@/app/components/ui/button';
import { Textarea } from '@/app/components/ui/textarea';
import { Card, CardContent } from '@/app/components/ui/card';
import { Label } from '@/app/components/ui/label';

export function UploadPage() {
  const navigate = useNavigate();
  const [cvText, setCvText] = useState('');
  const [preferences, setPreferences] = useState({
    industry: '',
    country: '',
    region: '',
    location: '',
    roleType: [] as string[],
    techStack: [] as string[],
    confidentSkills: [] as string[],
  });
  const [showAdvancedFilters, setShowAdvancedFilters] = useState(false);
  const [loading, setLoading] = useState(false);
  const [file, setFile] = useState<File | null>(null);

  const toggleArrayItem = (array: string[], item: string) => {
    return array.includes(item)
      ? array.filter((i) => i !== item)
      : [...array, item];
  };
  const handleAnalyze = async () => {
    setLoading(true);
    // Simulate processing
    await new Promise(resolve => setTimeout(resolve, 1500));

    // Store data in sessionStorage for the results page
    sessionStorage.setItem('cvData', cvText);
    sessionStorage.setItem('preferences', JSON.stringify(preferences));

    setLoading(false);
    navigate('/survey');
  };

  const handleFileUpload = (uploadedFile: File) => {
    setFile(uploadedFile);
    setCvText(`File: ${uploadedFile.name}`);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
      <div className="container mx-auto px-4 py-12">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center mb-4">
            <Briefcase className="w-12 h-12 text-indigo-600" />
          </div>
          <h1 className="text-4xl mb-3 text-gray-900">Career Path Analyzer</h1>
          <p className="text-xl text-gray-600">
            Discover personalized job recommendations with AI-powered spreading activation
          </p>
        </div>

        <div className="max-w-4xl mx-auto space-y-6">
          {/* CV Input Card - Replaced with Premium File Upload */}
          <div className="mb-6">
            <FileUpload onFileUpload={handleFileUpload} />
          </div>

          {/* Career Preferences Form */}
          <div className="bg-white rounded-2xl shadow-xl p-8">
            <h2 className="text-2xl font-semibold text-gray-800 mb-6">Career Preferences</h2>

            <div className="space-y-4">
              {/* Basic Filters */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Preferred Industry
                </label>
                <select
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent bg-white"
                  value={preferences.industry}
                  onChange={(e: ChangeEvent<HTMLSelectElement>) => setPreferences({ ...preferences, industry: e.target.value })}
                >
                  <option value="">Select an industry</option>
                  <option value="technology">Technology</option>
                  <option value="finance">Finance</option>
                  <option value="healthcare">Healthcare</option>
                  <option value="marketing">Marketing</option>
                  <option value="education">Education</option>
                  <option value="consulting">Consulting</option>
                  <option value="retail">Retail</option>
                  <option value="manufacturing">Manufacturing</option>
                </select>
              </div>

              {/* Location Filters Section */}
              <div className="border-t pt-4 mt-6">
                <h3 className="text-lg font-semibold text-gray-800 mb-4">Location Preferences</h3>

                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Country
                    </label>
                    <select
                      className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent bg-white"
                      value={preferences.country}
                      onChange={(e: ChangeEvent<HTMLSelectElement>) => setPreferences({ ...preferences, country: e.target.value, region: '' })}
                    >
                      <option value="">Any Country</option>
                      {countryOptions.map(country => (
                        <option key={country} value={country}>{country}</option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Region/State
                    </label>
                    <select
                      className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent bg-white"
                      value={preferences.region}
                      onChange={(e: ChangeEvent<HTMLSelectElement>) => setPreferences({ ...preferences, region: e.target.value })}
                      disabled={!preferences.country || !regionOptions[preferences.country]}
                    >
                      <option value="">Any Region</option>
                      {preferences.country && regionOptions[preferences.country]?.map(region => (
                        <option key={region} value={region}>{region}</option>
                      ))}
                    </select>
                  </div>
                </div>

                <div className="mt-4">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Specific City/Location (Optional)
                  </label>
                  <input
                    type="text"
                    placeholder="e.g., San Francisco, London, Remote"
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                    value={preferences.location}
                    onChange={(e: ChangeEvent<HTMLInputElement>) => setPreferences({ ...preferences, location: e.target.value })}
                  />
                </div>
              </div>

              {/* Role Type Filter */}
              <div className="border-t pt-4 mt-6">
                <h3 className="text-lg font-semibold text-gray-800 mb-4">Role Type</h3>
                <div className="flex flex-wrap gap-2">
                  {roleTypeOptions.map(type => (
                    <button
                      key={type}
                      onClick={() => setPreferences({
                        ...preferences,
                        roleType: toggleArrayItem(preferences.roleType, type)
                      })}
                      className={`px-4 py-2 rounded-lg font-medium transition-all ${preferences.roleType.includes(type)
                        ? 'bg-indigo-600 text-white'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                        }`}
                    >
                      {type}
                    </button>
                  ))}
                </div>
              </div>

              {/* Advanced Filters Toggle */}
              <div className="border-t pt-4 mt-6">
                <button
                  onClick={() => setShowAdvancedFilters(!showAdvancedFilters)}
                  className="flex items-center justify-between w-full text-left"
                >
                  <h3 className="text-lg font-semibold text-gray-800">Advanced Filters</h3>
                  <ChevronRight className={`w-5 h-5 text-gray-600 transition-transform ${showAdvancedFilters ? 'rotate-90' : ''}`} />
                </button>

                {showAdvancedFilters && (
                  <div className="mt-4 space-y-6">
                    {/* Technical Stack Filter */}
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-3">
                        Technical Stack / Programming Languages
                      </label>
                      <p className="text-xs text-gray-500 mb-3">Select the technologies you want to work with</p>
                      <div className="flex flex-wrap gap-2 max-h-64 overflow-y-auto p-2 border border-gray-200 rounded-lg">
                        {techStackOptions.map(tech => (
                          <button
                            key={tech}
                            onClick={() => setPreferences({
                              ...preferences,
                              techStack: toggleArrayItem(preferences.techStack, tech)
                            })}
                            className={`px-3 py-1.5 rounded-full text-sm font-medium transition-all ${preferences.techStack.includes(tech)
                              ? 'bg-purple-600 text-white'
                              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                              }`}
                          >
                            {tech}
                          </button>
                        ))}
                      </div>
                      {preferences.techStack.length > 0 && (
                        <p className="text-xs text-indigo-600 mt-2">
                          {preferences.techStack.length} selected
                        </p>
                      )}
                    </div>

                    {/* Confident Skills Filter */}
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-3">
                        Skills You're Confident In
                      </label>
                      <p className="text-xs text-gray-500 mb-3">Select skills where you have strong proficiency</p>
                      <div className="flex flex-wrap gap-2 max-h-64 overflow-y-auto p-2 border border-gray-200 rounded-lg">
                        {skillsOptions.map(skill => (
                          <button
                            key={skill}
                            onClick={() => setPreferences({
                              ...preferences,
                              confidentSkills: toggleArrayItem(preferences.confidentSkills, skill)
                            })}
                            className={`px-3 py-1.5 rounded-full text-sm font-medium transition-all ${preferences.confidentSkills.includes(skill)
                              ? 'bg-green-600 text-white'
                              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                              }`}
                          >
                            {skill}
                          </button>
                        ))}
                      </div>
                      {preferences.confidentSkills.length > 0 && (
                        <p className="text-xs text-green-600 mt-2">
                          {preferences.confidentSkills.length} selected
                        </p>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>

            <button
              onClick={handleAnalyze}
              disabled={!file || loading}
              className="w-full mt-6 bg-gradient-to-r from-indigo-600 to-purple-600 text-white py-4 rounded-lg font-semibold hover:from-indigo-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
            >
              {loading ? 'Analyzing CV...' : 'Next'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

interface FileUploadProps {
  onFileUpload: (file: File) => void;
}

function FileUpload({ onFileUpload }: FileUploadProps) {
  const [dragActive, setDragActive] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);

  const handleDrag = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      handleFile(file);
    }
  };

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file: File) => {
    const validTypes = [
      'application/pdf',
      'application/msword',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    ];

    if (validTypes.includes(file.type)) {
      setUploadedFile(file);
      onFileUpload(file);
    } else {
      alert('Please upload a PDF or Word document');
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Card className="border-2 border-dashed transition-all duration-300 hover:border-indigo-400 bg-white">
        <CardContent className="p-8">
          <form
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onSubmit={(e) => e.preventDefault()}
          >
            <input
              type="file"
              id="file-upload"
              className="hidden"
              accept=".pdf,.doc,.docx"
              onChange={handleChange}
            />

            <label
              htmlFor="file-upload"
              className={`flex flex-col items-center justify-center cursor-pointer transition-all duration-300 ${dragActive ? 'scale-105' : ''
                }`}
            >
              {uploadedFile ? (
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  className="flex flex-col items-center gap-4"
                >
                  <CheckCircle2 className="w-16 h-16 text-green-500" />
                  <div className="flex items-center gap-2 text-lg">
                    <FileText className="w-5 h-5" />
                    <span className="font-medium">{uploadedFile.name}</span>
                  </div>
                  <p className="text-sm text-gray-500">
                    {(uploadedFile.size / 1024).toFixed(2)} KB
                  </p>
                  <Button
                    type="button"
                    variant="outline"
                    onClick={(e) => {
                      e.preventDefault();
                      setUploadedFile(null);
                    }}
                  >
                    Change File
                  </Button>
                </motion.div>
              ) : (
                <div className="flex flex-col items-center gap-4">
                  <motion.div
                    animate={{ y: [0, -10, 0] }}
                    transition={{ duration: 2, repeat: Infinity }}
                  >
                    <Upload className="w-16 h-16 text-indigo-500" />
                  </motion.div>
                  <div className="text-center">
                    <p className="text-lg font-semibold mb-2">
                      Upload your CV/Resume
                    </p>
                    <p className="text-sm text-gray-500 mb-4">
                      Drag and drop or click to browse
                    </p>
                    <p className="text-xs text-gray-400">
                      Supported formats: PDF, DOC, DOCX
                    </p>
                  </div>
                </div>
              )}
            </label>
          </form>
        </CardContent>
      </Card>
    </motion.div>
  );
}

// Constants
const countryOptions = ['United States', 'United Kingdom', 'Canada', 'Germany', 'France', 'Australia', 'India', 'Singapore'];
const regionOptions: Record<string, string[]> = {
  'United States': ['California', 'New York', 'Texas', 'Washington', 'Massachusetts'],
  'United Kingdom': ['London', 'Manchester', 'Hampshire', 'West Midlands'],
  'Canada': ['Ontario', 'British Columbia', 'Quebec', 'Alberta'],
  'Germany': ['Berlin', 'Munich', 'Hamburg', 'Frankfurt'],
  'France': ['Paris', 'Lyon', 'Marseille', 'Toulouse'],
  'Australia': ['New South Wales', 'Victoria', 'Queensland'],
  'India': ['Karnataka', 'Maharashtra', 'Delhi', 'Telangana'],
  'Singapore': ['Central', 'East', 'West', 'North', 'North-East']
};
const roleTypeOptions = ['Full-time', 'Part-time', 'Contract', 'Freelance', 'Internship', 'Remote', 'Hybrid'];

const techStackOptions = [
  'JavaScript', 'Python', 'Java', 'C++', 'C#', 'TypeScript', 'Ruby', 'Go', 'Rust', 'PHP',
  'React', 'Angular', 'Vue.js', 'Node.js', 'Django', 'Flask', 'Spring Boot',
  'SQL', 'MongoDB', 'PostgreSQL', 'MySQL', 'Redis',
  'AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'Jenkins', 'Git',
  'TensorFlow', 'PyTorch', 'Scikit-learn', 'Pandas', 'NumPy'
];

const skillsOptions = [
  'Agile/Scrum', 'Project Management', 'Leadership', 'Communication', 'Team Collaboration',
  'Problem Solving', 'Critical Thinking', 'Data Analysis', 'UI/UX Design', 'API Design',
  'System Architecture', 'Database Design', 'DevOps', 'CI/CD', 'Testing/QA',
  'Machine Learning', 'Data Science', 'Cloud Computing', 'Microservices', 'REST APIs'
];
