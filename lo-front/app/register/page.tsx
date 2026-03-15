import { Manrope } from "next/font/google";

const manrope = Manrope({
  subsets: ["latin"],
  variable: "--font-manrope",
});

export default function RegisterPage() {
  return (
    <div className={`${manrope.variable} font-sans flex w-full h-screen`}>
      
      {/* Top-left logo + title */}
      <div className="flex items-center absolute top-2 left-1 z-20">
        <img src="/assets/learningos-logo.png" alt="Logo" className="w-10 h-10 rounded-full"/>
        <h1 className="text-xl font-bold ml-2">LearningOS</h1>
      </div>

      {/* Left Panel */}
      <div className="hidden lg:flex w-1/2 bg-white border-r p-16 flex-col justify-center relative">
        <h1 className="text-5xl font-black leading-tight mb-6">
          Unlock Your Potential With AI
        </h1>
        <p className="top-3 text-[#4C669A] text-lg mb-10 relative z-10">
          Join us to master new skills with personalized paths and get started for free today.
        </p>
        <img
          src="/assets/ai-illustration.png"
          className="w-[500px] absolute bottom-0 left-69"
          alt="AI"
        />
      </div>

      {/* Right Panel */}
      <div className="w-full lg:w-1/2 flex flex-col items-center justify-center bg-gray-100 relative">
        <div className="w-[420px]">

          {/* Already have an account */}
          <div className="absolute top-4 right-4 text-right z-20">
            <span className="text-gray-500 text-sm">Already have an account? </span>
            <a href="#" className="text-blue-600 font-bold text-sm ml-1">Log in</a>
          </div>

          {/* Create Account Header */}
          <div className="flex items-center gap-3 mb-2 mt-12">
            <h2 className="text-4xl font-bold">Create your account</h2>
          </div>
          <p className="text-gray-500 mb-8">
            Start your personalized learning journey today.
          </p>

          {/* Social Login */}
          <div className="grid grid-cols-2 gap-4 mb-6">
            <button className="border h-12 rounded-lg flex items-center justify-center gap-2 bg-white">
              <img src="/assets/google-logo.png" alt="Google Logo" className="w-5 h-5"/>
              Google
            </button>
            <button className="border h-12 rounded-lg flex items-center justify-center gap-2 bg-white">
              <img src="/assets/microsoft-logo.png" alt="Microsoft Logo" className="w-5 h-5"/>
              Microsoft
            </button>
          </div>

          <div className="flex items-center">
  {/* Left Line */}
  <div className="flex-grow h-px bg-gray-200"></div>

  {/* YOUR TEXT GOES HERE */}
  <span className="flex-shrink px-4 text-xs font-medium text-gray-400 uppercase tracking-widest">
    OR REGISTER WITH EMAIL
  </span>

  {/* Right Line */}
  <div className="flex-grow h-px bg-gray-200"></div>
</div>

          {/* Form */}
          <form className="space-y-4 mt-2">

            {/* Name Fields */}
            <div className="flex gap-4">
              <div className="flex flex-col flex-1">
                <label className="text-sm font-semibold mb-1">First Name</label>
                <input className="border rounded-lg h-12 px-4 w-full" placeholder="Jane"/>
              </div>
              <div className="flex flex-col flex-1">
                <label className="text-sm font-semibold mb-1">Last Name</label>
                <input className="border rounded-lg h-12 px-4 w-full" placeholder="Doe"/>
              </div>
            </div>

            {/* Email Field */}
            <div className="relative">
              <label className="text-sm font-semibold mb-1">Work Email</label>
              <input
                type="email"
                placeholder="jane@company.com"
                className="w-full h-12 pl-11 pr-4 rounded-xl  border border-slate-200 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none transition-all "
              />
              
              <img
                src="/assets/email-icon.png"
                alt="Email Icon"
                className="absolute left-3 top-1/2 translate-y-1/7 w-5 h-5 "
              />
            </div>

            {/* Password Field */}
            <div className="relative">
              <label className="text-sm font-semibold mb-1">Password</label>
              <input
                type="password"
                placeholder="Min. 8 characters"
                className="w-full h-12 pl-11 pr-4 rounded-xl  border border-slate-200  focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none transition-all " />
              
              
              <img
                src="/assets/lock-icon.png"
                alt="Lock Icon"
                className="absolute left-3 top-1/2 translate-y-1/14 w-5 h-5"
              />
            </div>

            {/* Password Strength */}
            <div className="flex gap-1 mt-1">
              <div className="h-1 flex-1 rounded-full bg-emerald-500"></div>
              <div className="h-1 flex-1 rounded-full bg-emerald-500"></div>
              <div className="h-1 flex-1 rounded-full bg-gray-200"></div>
              <div className="h-1 flex-1 rounded-full bg-gray-200"></div>
            </div>

            {/* Terms Checkbox */}
            <div className="flex items-center gap-2 mt-2">
              <input type="checkbox" id="terms" className="w-4 h-4 text-blue-600 border-gray-300 rounded"/>
              <label htmlFor="terms" className="text-sm text-gray-500">
                I agree to the <a href="#" className="text-blue-600">Terms</a> and <a href="#" className="text-blue-600">Privacy Policy</a>.
              </label>
            </div>

            <button className="w-full h-12 bg-blue-600 text-white rounded-lg font-semibold flex items-center justify-center gap-2">
              Create Account
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}