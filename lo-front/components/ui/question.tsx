export default function Question({question}: {question: string}) {
  return (
    <div className="flex gap-4 bg-[#EDF9FF] p-4 rounded-lg">
  <label className="flex items-center gap-3 cursor-pointer">
  <input
    type="radio"
    name="option"
    value={question}
    className="peer sr-only"
  />

  {/* Circle */}
  <div className="w-5 h-5 rounded-full border-2 border-gray-400 
                  flex items-center justify-center
                  peer-checked:border-blue-500">
    
    <div className="w-2.5 h-2.5 rounded-full bg-blue-500 
                    opacity-0 peer-checked:opacity-100 
                    transition" />
  </div>

  {/* Text */}
  <span>{question}</span>
</label>
</div>
  );
}