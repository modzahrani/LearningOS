type QuestionProps = {
  question: string;
  onSelect?: () => void;
  selected?: boolean;
  disabled?: boolean;
};

export default function Question({
  question,
  onSelect,
  selected = false,
  disabled = false,
}: QuestionProps) {
  return (
    <button
      type="button"
      onClick={onSelect}
      disabled={disabled}
      className={[
        "w-full text-left px-4 py-3 rounded-lg border transition",
        "hover:border-blue-400 hover:bg-blue-50",
        "disabled:opacity-60 disabled:hover:border-gray-200 disabled:hover:bg-white",
        selected ? "border-blue-500 bg-blue-50" : "border-gray-200 bg-white",
      ].join(" ")}
    >
      <p className="text-sm text-[#4C669A]">{question}</p>
    </button>
  );
}
