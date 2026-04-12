import { Button } from "@/components/ui/button"

export function Primarybtn({ button }: { button: string }) {
  return <Button className="bg-blue-500 w-[159px] hover:bg-blue-600 text-white">{button}</Button>
}
