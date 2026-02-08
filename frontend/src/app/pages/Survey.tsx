import { useState, ChangeEvent } from 'react';
import { useFormContext } from '@/app/context/FormContext';
import { motion, AnimatePresence } from 'motion/react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/app/components/ui/card';
import { Button } from '@/app/components/ui/button';
import { Label } from '@/app/components/ui/label';
import { Progress } from '@/app/components/ui/progress';
import { Textarea } from '@/app/components/ui/textarea';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { useNavigate } from 'react-router';

const SURVEY_QUESTIONS = [
    {
        id: 'engagement',
        question: 'What activities make you feel most engaged or absorbed when you’re doing them?',
        placeholder: 'Think about moments when you lose track of time, whether in school, work, or daily life.',
    },
    {
        id: 'problem_solving_motivation',
        question: 'What types of problems do you naturally notice or feel motivated to solve?',
        placeholder: 'These could involve people, processes, ideas, organization, or communication.',
    },
    {
        id: 'ideal_environment',
        question: 'What kind of work environment do you think you would function best in?',
        placeholder: 'Consider structure vs flexibility, teamwork vs independence, pace, or pressure level.',
    },
    {
        id: 'helped_skills',
        question: 'What skills do people often come to you for help with?',
        placeholder: 'This might include explaining things, organizing tasks, listening, planning, or creative input.',
    },
    {
        id: 'proud_experiences',
        question: 'Which past experiences left you feeling proud or satisfied, and why?',
        placeholder: 'Focus on what you personally contributed rather than external rewards or grades.',
    },
    {
        id: 'desired_impact',
        question: 'What kind of impact would you like your work to have on others or society?',
        placeholder: 'You may think about helping individuals, improving systems, creating value, or influencing change.',
    },
    {
        id: 'energy_drainers',
        question: 'What types of tasks drain your energy or you strongly dislike doing?',
        placeholder: 'Be honest—this helps rule out paths that are likely a poor fit.',
    },
    {
        id: 'ideal_workday',
        question: 'When you imagine your ideal workday, what does it roughly look like?',
        placeholder: 'Describe how you spend your time, who you interact with, and what you focus on.',
    },
    {
        id: 'career_constraints',
        question: 'What constraints matter most to you when choosing a career path?',
        placeholder: 'Examples include income stability, work-life balance, location, growth opportunities, or job security.',
    },
    {
        id: 'future_exploration',
        question: 'If you had to choose one direction to explore for the next year, what would you want to learn or try first?',
        placeholder: 'This could be an industry, role type, skill area, or real-world exposure like internships or volunteering.',
    },
];

export function SurveyPage() {
    const navigate = useNavigate();
    const { state, updateSurveyAnswer, setResults } = useFormContext();
    const { surveyAnswers: answers } = state;
    const [currentQuestion, setCurrentQuestion] = useState(0);
    const [direction, setDirection] = useState(0);
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const progress = ((currentQuestion + 1) / SURVEY_QUESTIONS.length) * 100;

    const handleAnswer = (value: string) => {
        updateSurveyAnswer(SURVEY_QUESTIONS[currentQuestion].id, value);
    };

    const handleSubmit = async () => {
        setIsSubmitting(true);
        setError(null);

        try {
            const formData = new FormData();

            // 1. Add CV File if exists
            if (state.cvFile) {
                formData.append('cv_file', state.cvFile);
            }

            // 2. Add CV Text metadata
            formData.append('cv_text', state.cvText);

            // 3. Add Preferences
            formData.append('preferences', JSON.stringify(state.preferences));

            // 4. Add Survey Answers
            formData.append('survey_answers', JSON.stringify(state.surveyAnswers));

            // Standard backend call
            const response = await fetch('/api/v1/assessments', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error('Failed to submit assessment. Please try again.');
            }

            const data = await response.json();

            // Success: save results and navigate to results
            if (data.results) {
                setResults(data.results);
            }
            navigate('/results');
        } catch (err) {
            console.error('Submission error:', err);
            setError(err instanceof Error ? err.message : 'An unexpected error occurred');
        } finally {
            setIsSubmitting(false);
        }
    };

    const goToNext = () => {
        if (currentQuestion < SURVEY_QUESTIONS.length - 1) {
            setDirection(1);
            setCurrentQuestion(prev => prev + 1);
        } else if (answers[SURVEY_QUESTIONS[currentQuestion].id]) {
            handleSubmit(); /* 
            The Frontend "Sends"
            This function packages all data from FormContext.tsx into a FormData object and 'posts' 
            it to a specific address (an endpoint like /api/analyze). 
        
            The Backend "Receives"
            We write a function in the backend that is "listening" for that specific package using the endpoint (e.g. /api/analyze).
        
            */

        }
    };

    const goToPrevious = () => {
        if (currentQuestion > 0) {
            setDirection(-1);
            setCurrentQuestion(prev => prev - 1);
        }
    };

    const currentAnswer = answers[SURVEY_QUESTIONS[currentQuestion].id];
    const isLastQuestion = currentQuestion === SURVEY_QUESTIONS.length - 1;

    return (
        <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 py-12 px-4">
            <div className="max-w-3xl mx-auto">
                <Button
                    variant="ghost"
                    onClick={() => navigate('/')}
                    className="mb-6 bg-indigo-600 hover:bg-indigo-700 text-white font-bold px-8 py-4 rounded-2xl shadow-lg shadow-indigo-200 gap-2 transition-all"
                >
                    <ChevronLeft className="w-5 h-5" />
                    Back to Preferences
                </Button>
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5 }}
                >
                    <Card className="shadow-xl">
                        <CardHeader className="space-y-4">
                            <div className="flex items-center justify-between">
                                <div>
                                    <CardTitle className="text-2xl font-bold">Career Assessment</CardTitle>
                                    <CardDescription>Tailoring your trajectory</CardDescription>
                                </div>
                                <span className="text-sm font-semibold py-1 px-3 bg-blue-50 text-blue-600 rounded-full">
                                    Question {currentQuestion + 1} of {SURVEY_QUESTIONS.length}
                                </span>
                            </div>
                            <Progress value={progress} className="h-2" />
                        </CardHeader>
                        <CardContent className="min-h-[450px] flex flex-col pt-6">
                            <AnimatePresence mode="wait" custom={direction}>
                                <motion.div
                                    key={currentQuestion}
                                    custom={direction}
                                    initial={{ opacity: 0, x: direction > 0 ? 50 : -50 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    exit={{ opacity: 0, x: direction > 0 ? -50 : 50 }}
                                    transition={{ duration: 0.2 }}
                                    className="flex-1"
                                >
                                    <h3 className="text-xl font-bold text-gray-900 mb-8">
                                        {SURVEY_QUESTIONS[currentQuestion].question}
                                    </h3>

                                    <Textarea
                                        value={currentAnswer || ''}
                                        onChange={(e: ChangeEvent<HTMLTextAreaElement>) => handleAnswer(e.target.value)}
                                        placeholder={SURVEY_QUESTIONS[currentQuestion].placeholder}
                                        className="min-h-[200px] text-lg p-6 rounded-2xl border-2 border-gray-100 focus:border-indigo-500 transition-all bg-white shadow-inner"
                                        disabled={isSubmitting}
                                    />
                                </motion.div>
                            </AnimatePresence>

                            {error && (
                                <p className="mt-4 text-red-500 text-center font-semibold">
                                    {error}
                                </p>
                            )}

                            <div className="flex items-center justify-between mt-12 pt-8 border-t border-gray-100">
                                <Button
                                    variant="ghost"
                                    onClick={goToPrevious}
                                    disabled={currentQuestion === 0 || isSubmitting}
                                    className="gap-2 font-bold px-6 py-6 rounded-2xl"
                                >
                                    <ChevronLeft className="w-5 h-5" />
                                    Back
                                </Button>
                                <Button
                                    onClick={goToNext}
                                    disabled={!currentAnswer || isSubmitting}
                                    className="bg-indigo-600 hover:bg-indigo-700 text-white font-bold px-10 py-6 rounded-2xl shadow-lg shadow-indigo-200 gap-2 transition-all relative overflow-hidden"
                                >
                                    {isSubmitting ? (
                                        <span className="flex items-center gap-2">
                                            <motion.div
                                                animate={{ rotate: 360 }}
                                                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                                                className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full"
                                            />
                                            Analyzing...
                                        </span>
                                    ) : (
                                        <>
                                            {isLastQuestion ? 'Reveal Results' : 'Continue'}
                                            {!isLastQuestion && <ChevronRight className="w-5 h-5" />}
                                        </>
                                    )}
                                </Button>
                            </div>
                        </CardContent>
                    </Card>
                </motion.div>
            </div>
        </div>
    );
}
