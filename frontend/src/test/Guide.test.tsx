import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { Guide } from "../components/Guide";

describe("Guide", () => {
  it("keeps every page's detail collapsed until it is opened", () => {
    render(<Guide onNavigate={vi.fn()} />);
    expect(
      screen.getByRole("button", { name: /open workers/i }),
    ).not.toBeVisible();
    expect(screen.getByText("Workers")).toBeVisible();
  });

  it("hands the reader off to the page they opened", async () => {
    const onNavigate = vi.fn();
    render(<Guide onNavigate={onNavigate} />);

    await userEvent.click(screen.getByText("Workers"));
    await userEvent.click(
      screen.getByRole("button", { name: /open workers/i }),
    );

    expect(onNavigate).toHaveBeenCalledWith("Workers");
  });

  it("reveals a stage's reasoning from its question mark", async () => {
    render(<Guide onNavigate={vi.fn()} />);
    const probe = screen.getByText("Probe").closest("li");
    expect(probe).not.toBeNull();

    const why = within(probe as HTMLElement).getByLabelText(
      /why it works this way/i,
    );
    expect(screen.queryByText(/fails here in seconds/i)).not.toBeVisible();

    await userEvent.click(why);
    expect(screen.getByText(/fails here in seconds/i)).toBeVisible();
  });

  it("states the limit alongside what a page can do", async () => {
    render(<Guide onNavigate={vi.fn()} />);
    await userEvent.click(screen.getByText("Assistant"));
    expect(
      screen.getByText(/eight tool calls and 90 seconds per turn/i),
    ).toBeVisible();
  });
});
