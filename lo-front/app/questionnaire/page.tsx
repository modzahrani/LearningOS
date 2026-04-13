"use client";
import "./styles.css";
import Question from "@/components/ui/question";
import { Bot } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import {
  QuizAnswerResponse,
  QuizQuestion,
  QuizStartResponse,
  answerQuiz,
  resumeQuiz,
  startQuiz,
} from "@/api/quizProvider";

export default function Questionnaire() {
  const router = useRouter();
  const QUIZ_ID_STORAGE_KEY = "learningos_quiz_id";
  const [quizId, setQuizId] = useState<string | null>(null);
  const [question, setQuestion] = useState<QuizQuestion | null>(null);
  const [questionNumber, setQuestionNumber] = useState(0);
  const [totalQuestions, setTotalQuestions] = useState(0);
  const [progress, setProgress] = useState(0);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [finished, setFinished] = useState(false);
  const [lastCorrect, setLastCorrect] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);

  const canSubmit = useMemo(
    () => selectedIndex !== null && !submitting && !loading && !finished,
    [selectedIndex, submitting, loading, finished]
  );

  const handleStartQuiz = async () => {
    setLoading(true);
    setError(null);
    setFinished(false);
    setLastCorrect(null);
    setSelectedIndex(null);

    try {
      const response = await startQuiz();
      const data: QuizStartResponse = response.data;
      localStorage.setItem(QUIZ_ID_STORAGE_KEY, data.quiz_id);
      setQuizId(data.quiz_id);
      setQuestion(data.question);
      setQuestionNumber(data.question_number);
      setTotalQuestions(data.total_questions);
      setProgress(data.progress_percent);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start quiz");
    } finally {
      setLoading(false);
    }
  };

  const submitAnswer = async () => {
    if (selectedIndex === null || !quizId || submitting) return;

    setSubmitting(true);
    setError(null);

    try {
      const response = await answerQuiz({
        quiz_id: quizId,
        answer_index: selectedIndex,
      });
      const data: QuizAnswerResponse = response.data;
      setLastCorrect(data.correct);
      setFinished(data.finished);
      setQuestionNumber(data.question_number);
      setTotalQuestions(data.total_questions);
      setProgress(data.progress_percent);
      setSelectedIndex(null);

      if (data.question) {
        setQuestion(data.question);
      } else {
        setQuestion(null);
        localStorage.removeItem(QUIZ_ID_STORAGE_KEY);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to submit answer");
    } finally {
      setSubmitting(false);
    }
  };

  useEffect(() => {
    const bootstrapQuiz = async () => {
      setLoading(true);
      setError(null);
      try {
        const resumed = await resumeQuiz();
        const data = resumed.data;
        localStorage.setItem(QUIZ_ID_STORAGE_KEY, data.quiz_id);
        setQuizId(data.quiz_id);
        setQuestion(data.question);
        setQuestionNumber(data.question_number);
        setTotalQuestions(data.total_questions);
        setProgress(data.progress_percent);
        setFinished(false);
        setLastCorrect(null);
        setSelectedIndex(null);
      } catch {
        await handleStartQuiz();
      } finally {
        setLoading(false);
      }
    };

    bootstrapQuiz();
  }, []);

  return (
    <div className="flex flex-col gap-2 justify-center items-center min-h-screen px-12 bg-[#F6F6F8]">
      
      <div className="w-full max-w-[1300px]">
        <h1 className="text-3xl font-bold">AI Knowledge Calibration</h1>
        <p className="text-lg text-[#4C669A] mt-1">
          Answer these questions to help our AI tailor the curriculum to your expertise. Your answers determine your starting level.
        </p>
      </div>

      <div className="flex gap-6 items-start w-full max-w-[1300px]">

        {/* Main question card */}
        <section className="flex flex-col gap-4 flex-1 bg-white rounded-lg shadow-lg p-6 min-h-[520px]">
          <header className="flex justify-between items-center">
            <h2 className="text-md font-semibold">
              {loading ? "Loading..." : `Question ${questionNumber} of ${totalQuestions}`}
            </h2>
            <p className="text-sm text-[#4C669A]">{progress}% Completed</p>
          </header>

          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${progress}%` }} />
          </div>

          <h1 className="text-xl font-bold">
            {question?.text ?? (loading ? "Loading question..." : "No question available")}
          </h1>

          <div className="flex flex-col gap-4">
            {question?.options?.map((option, index) => (
              <Question
                key={option}
                question={option}
                selected={selectedIndex === index}
                disabled={loading || submitting || finished}
                onSelect={() => setSelectedIndex(index)}
              />
            ))}
            {!loading && !question && (
              <p className="text-sm text-red-500">No question data returned.</p>
            )}
          </div>

          <div className="flex justify-between items-center mt-auto pt-4">
            <div className="text-sm text-[#4C669A]">
              {error && <span className="text-red-500">{error}</span>}
              {!error && lastCorrect === true && "Correct!"}
              {!error && lastCorrect === false && "Not quite — try the next one."}
              {!error && finished && "Quiz complete. You can restart."}
            </div>
            {finished ? (
              <button
                type="button"
                onClick={handleStartQuiz}
                className="bg-blue-500 w-[159px] hover:bg-blue-600 text-white rounded-md h-9 px-4"
              >
                Restart Quiz
              </button>
            ) : (
              <button
                type="button"
                onClick={submitAnswer}
                disabled={!canSubmit}
                className={[
                  "w-[159px] rounded-md h-9 px-4 text-white transition",
                  canSubmit
                    ? "bg-blue-500 hover:bg-blue-600"
                    : "bg-blue-300 cursor-not-allowed",
                ].join(" ")}
              >
                {submitting ? "Submitting..." : "Next Question"}
              </button>
            )}
          </div>
        </section>

        {/* AI Calibration sidebar + button */}
        <div className="flex flex-col gap-2 w-[300px] shrink-0">
          <section className="flex flex-col gap-4 bg-white rounded-lg shadow-lg p-6">
            <header className="flex items-start gap-2">
              <Bot className="text-[#4C669A] w-[44.5px] h-[44.5px] shrink-0" />
              <div className="flex flex-col gap-1">
                <h2 className="text-sm font-semibold">AI Calibration</h2>
                <p className="text-sm text-[#4C669A]">Analyzing your responses...</p>
              </div>
            </header>

            <p className="text-sm text-[#4C669A] leading-relaxed">
              Based on your previous answers, we&apos;ve adjusted this question to test your fundamental understanding of ML categories.
            </p>

            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <p className="text-sm text-[#4C669A]">AI adjusting in real-time</p>
            </div>
          </section>

          <button
            className="text-xs text-[#4C669A] text-center"
            onClick={() => router.push("/path-select")}
          >
            back to path selection?
          </button>
        </div>

      </div>
    </div>
  );
}
