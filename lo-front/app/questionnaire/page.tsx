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

      if (data.finished) {
        localStorage.removeItem(QUIZ_ID_STORAGE_KEY);
        router.push("/lessons");
        return;
      }

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
    <main className="min-h-screen bg-[radial-gradient(circle_at_top_right,rgba(37,99,235,0.16),transparent_28%),linear-gradient(180deg,var(--background),color-mix(in_oklab,var(--muted)_68%,transparent))] px-4 py-8 md:px-8">
      <div className="mx-auto flex w-full max-w-[1300px] flex-col gap-6">
        <div>
        <h1 className="text-3xl font-bold text-foreground md:text-4xl">AI Knowledge Calibration</h1>
        <p className="mt-2 max-w-4xl text-base text-muted-foreground md:text-lg">
          Answer these questions to help our AI tailor the curriculum to your expertise. Your answers determine your starting level.
        </p>
      </div>

      <div className="flex w-full flex-col gap-6 xl:flex-row xl:items-start">

        {/* Main question card */}
        <section className="flex min-h-[520px] flex-1 flex-col gap-5 rounded-[28px] border border-border bg-card/95 p-6 shadow-[0_24px_80px_rgba(15,23,42,0.10)] backdrop-blur md:p-8">
          <header className="flex justify-between items-center">
            <h2 className="text-sm font-semibold text-foreground md:text-base">
              {loading ? "Loading..." : `Question ${questionNumber} of ${totalQuestions}`}
            </h2>
            <p className="text-sm text-blue-600 dark:text-blue-400">{progress}% Completed</p>
          </header>

          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${progress}%` }} />
          </div>

          <h1 className="text-xl font-bold text-foreground md:text-2xl">
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

          <div className="mt-auto flex flex-col gap-4 border-t border-border pt-4 md:flex-row md:items-center md:justify-between">
            <div className="text-sm text-muted-foreground">
              {error && <span className="text-red-500">{error}</span>}
              {!error && lastCorrect === true && "Correct!"}
              {!error && lastCorrect === false && "Not quite — try the next one."}
              {!error && finished && "Quiz complete. You can restart."}
            </div>
            {finished ? (
              <button
                type="button"
                onClick={handleStartQuiz}
                className="h-11 rounded-xl bg-blue-600 px-5 text-sm font-semibold text-white transition hover:bg-blue-700"
              >
                Restart Quiz
              </button>
            ) : (
              <button
                type="button"
                onClick={submitAnswer}
                disabled={!canSubmit}
                className={[
                  "h-11 rounded-xl px-5 text-sm font-semibold text-white transition md:min-w-[159px]",
                  canSubmit
                    ? "bg-blue-600 hover:bg-blue-700"
                    : "cursor-not-allowed bg-blue-300 dark:bg-blue-900/50",
                ].join(" ")}
              >
                {submitting ? "Submitting..." : "Next Question"}
              </button>
            )}
          </div>
        </section>

        {/* AI Calibration sidebar + button */}
        <div className="flex w-full shrink-0 flex-col gap-3 xl:w-[320px]">
          <section className="flex flex-col gap-4 rounded-[28px] border border-border bg-card/95 p-6 shadow-[0_24px_80px_rgba(15,23,42,0.10)] backdrop-blur">
            <header className="flex items-start gap-2">
              <Bot className="h-[44.5px] w-[44.5px] shrink-0 text-blue-600 dark:text-blue-400" />
              <div className="flex flex-col gap-1">
                <h2 className="text-sm font-semibold text-foreground">AI Calibration</h2>
                <p className="text-sm text-blue-600 dark:text-blue-400">Analyzing your responses...</p>
              </div>
            </header>

            <p className="text-sm leading-relaxed text-muted-foreground">
              Based on your previous answers, we&apos;ve adjusted this question to test your fundamental understanding of ML categories.
            </p>

            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <p className="text-sm text-muted-foreground">AI adjusting in real-time</p>
            </div>
          </section>

          <button
            className="text-center text-xs font-medium text-blue-600 transition hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300"
            onClick={() => router.push("/path-select")}
          >
            Back to path selection
          </button>
        </div>

      </div>
    </div>
    </main>
  );
}
