import { aiDevtoolsPlugin } from "@tanstack/react-ai-devtools"
import { TanStackDevtools } from "@tanstack/react-devtools"
import { HeadContent, Scripts, createRootRoute } from "@tanstack/react-router"
import { TanStackRouterDevtoolsPanel } from "@tanstack/react-router-devtools"
import { ThemeProvider } from "next-themes"

import appCss from "../styles.css?url"

/**
 * The devtools let settings saved in the browser override their `config`,
 * so a `triggerHidden: false` saved before the button was hidden would keep
 * showing it. Drop that saved value so `config` decides.
 */
if (typeof window !== "undefined") {
  try {
    const key = "tanstack_devtools_settings"
    const saved: unknown = JSON.parse(localStorage.getItem(key) ?? "null")
    if (saved && typeof saved === "object" && "triggerHidden" in saved) {
      const { triggerHidden: _, ...rest } = saved
      localStorage.setItem(key, JSON.stringify(rest))
    }
  } catch {
    // Storage is blocked or the value isn't JSON: nothing to fix.
  }
}

export const Route = createRootRoute({
  head: () => ({
    meta: [
      {
        charSet: "utf-8"
      },
      // Hugging Face rejects model downloads whose Referer is a *.workers.dev page.
      {
        name: "referrer",
        content: "no-referrer"
      },
      {
        name: "viewport",
        content: "width=device-width, initial-scale=1"
      },
      {
        title: "lofi-agent"
      }
    ],
    links: [
      {
        rel: "stylesheet",
        href: appCss
      },
      // Add to Home Screen: name, start page and icons come from the manifest.
      // An installed app also keeps its storage (the downloaded model) more
      // reliably than a site in Safari.
      { rel: "manifest", href: "/manifest.webmanifest" },
      { rel: "icon", href: "/icon.svg", type: "image/svg+xml" },
      { rel: "apple-touch-icon", href: "/apple-touch-icon.png" }
    ]
  }),
  shellComponent: RootDocument
})

function RootDocument({ children }: { children: React.ReactNode }) {
  return (
    // next-themes sets class + data-theme on <html> before paint, so React's
    // hydration sees different attributes than the server sent.
    <html lang="en" suppressHydrationWarning>
      <head>
        <HeadContent />
        {/* The browser bar and the Home Screen app's status bar match the
            page. Written here: the head manager keeps one tag per name. */}
        <meta
          name="theme-color"
          content="#f5f5f5"
          media="(prefers-color-scheme: light)"
        />
        <meta
          name="theme-color"
          content="#09090b"
          media="(prefers-color-scheme: dark)"
        />
      </head>
      <body className="bg-background text-foreground">
        {/* HeroUI reads both the class and data-theme attributes */}
        <ThemeProvider
          attribute={["class", "data-theme"]}
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          {children}
        </ThemeProvider>
        {/* TanStack devtools, in development only (the Vite plugin removes
            them from builds), with no button: open them with Ctrl + ~.
            Chats run in the page, so the AI panel sees each run's model
            calls and tool calls with no server bus. */}
        <TanStackDevtools
          config={{ triggerHidden: true }}
          plugins={[
            aiDevtoolsPlugin(),
            {
              name: "TanStack Router",
              render: <TanStackRouterDevtoolsPanel />
            }
          ]}
        />
        <Scripts />
      </body>
    </html>
  )
}
