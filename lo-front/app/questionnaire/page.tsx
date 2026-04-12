import "./styles.css"
import Question from "@/components/ui/question";
import { Primarybtn } from "@/components/ui/primaryBtn";
import { Bot } from "lucide-react";

export default function Questionnaire() {
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
        <section className="flex flex-col gap-4 flex-1 bg-white rounded-lg shadow-lg p-6">
          <header className="flex justify-between items-center">
            <h2 className="text-md font-semibold">Question 1 of 10</h2>
            <p className="text-sm text-[#4C669A]">30% Completed</p>
          </header>

          <div className="progress-track">
            <div className="progress-fill" style={{ width: '30%' }} />
          </div>

          <h1 className="text-xl font-bold">
            What is the primary difference between supervised and unsupervised learning?
          </h1>

          <div className="flex flex-col gap-4">
            <Question question="A) Supervised learning uses labeled data, while unsupervised learning uses unlabeled data." />
            <Question question="B) Supervised learning is faster than unsupervised learning." />
            <Question question="C) Unsupervised learning always yields more accurate results." />
            <Question question="D) There is no difference between them two." />
          </div>

          <div className="flex justify-between items-center mt-auto pt-4">
            <div></div>
            <Primarybtn button="Next Question" />
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
              Based on your previous answers, we've adjusted this question to test your fundamental understanding of ML categories.
            </p>

            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <p className="text-sm text-[#4C669A]">AI adjusting in real-time</p>
            </div>
          </section>

          <button className="text-xs text-[#4C669A] text-center">
            back to path selection?
          </button>
        </div>

      </div>
    </div>
  );
}