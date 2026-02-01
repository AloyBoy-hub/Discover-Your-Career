import React, { createContext, useContext, useState, ReactNode } from 'react';

export interface CareerPreferences {
    industry: string;
    country: string;
    region: string;
    location: string;
    roleType: string[];
    techStack: string[];
    confidentSkills: string[];
}

export interface JobResult {
    id: string;
    title: string;
    company: string;
    location: string;
    salaryRange: string;
    description: string;
    matchScore: number;
    skillsRequired: string[];
}

export interface FormState {
    cvFile: File | null;
    cvText: string;
    preferences: CareerPreferences;
    surveyAnswers: Record<string, string>;
    results: JobResult[] | null;
}

interface FormContextType {
    state: FormState;
    setCvFile: (file: File | null) => void;
    setCvText: (text: string) => void;
    setPreferences: (prefs: CareerPreferences) => void;
    updatePreference: <K extends keyof CareerPreferences>(key: K, value: CareerPreferences[K]) => void;
    setSurveyAnswers: (answers: Record<string, string>) => void;
    updateSurveyAnswer: (questionId: string, answer: string) => void;
    setResults: (results: JobResult[] | null) => void;
    resetForm: () => void;
}

const initialPreferences: CareerPreferences = {
    industry: '',
    country: '',
    region: '',
    location: '',
    roleType: [],
    techStack: [],
    confidentSkills: [],
};

const initialState: FormState = {
    cvFile: null,
    cvText: '',
    preferences: initialPreferences,
    surveyAnswers: {},
    results: null,
};

const FormContext = createContext<FormContextType | undefined>(undefined);

export function FormProvider({ children }: { children: ReactNode }) {
    const [state, setState] = useState<FormState>(initialState);

    const setCvFile = (file: File | null) => {
        setState(prev => ({ ...prev, cvFile: file }));
    };

    const setCvText = (text: string) => {
        setState(prev => ({ ...prev, cvText: text }));
    };

    const setPreferences = (preferences: CareerPreferences) => {
        setState(prev => ({ ...prev, preferences }));
    };

    const updatePreference = <K extends keyof CareerPreferences>(key: K, value: CareerPreferences[K]) => {
        setState(prev => ({
            ...prev,
            preferences: {
                ...prev.preferences,
                [key]: value
            }
        }));
    };

    const setSurveyAnswers = (surveyAnswers: Record<string, string>) => {
        setState(prev => ({ ...prev, surveyAnswers }));
    };

    const updateSurveyAnswer = (questionId: string, answer: string) => {
        setState(prev => ({
            ...prev,
            surveyAnswers: {
                ...prev.surveyAnswers,
                [questionId]: answer
            }
        }));
    };

    const setResults = (results: JobResult[] | null) => {
        setState(prev => ({ ...prev, results }));
    };

    const resetForm = () => {
        setState(initialState);
    };

    return (
        <FormContext.Provider value={{
            state,
            setCvFile,
            setCvText,
            setPreferences,
            updatePreference,
            setSurveyAnswers,
            updateSurveyAnswer,
            setResults,
            resetForm
        }}>
            {children}
        </FormContext.Provider>
    );
}

export function useFormContext() {
    const context = useContext(FormContext);
    if (context === undefined) {
        throw new Error('useFormContext must be used within a FormProvider');
    }
    return context;
}
