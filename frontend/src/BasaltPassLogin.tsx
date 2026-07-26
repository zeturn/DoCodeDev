import React, { useState } from 'react'
import { Card, Button, Stack } from '@zeturn/watercolor-react'
import { beginLogin } from './auth'

interface Props {
  app?: string
  brand?: string
  description?: string
}

export default function BasaltPassLogin({ app = 'app', brand = 'BasaltPass', description }: Props) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const handleLogin = async () => {
    setBusy(true)
    setError('')
    try {
      await beginLogin(app)
    } catch (err: any) {
      setBusy(false)
      setError(err.message)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-6" style={{ background: '#fbfbfd' }}>
      <Card variant="minimal" style={{ maxWidth: 360, width: '100%', background: 'var(--wc-surface-raised)' }}>
        <Stack gap="md" align="stretch">
          <div className="text-center">
            <div className="mx-auto mb-4 grid h-12 w-12 place-items-center rounded-2xl"
                 style={{ background: 'var(--wc-accent)', color: 'var(--wc-text-on-accent)' }}>
              <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                <path d="M12 3l8 4.5v9L12 21l-8-4.5v-9L12 3z" />
                <path d="M12 12l8-4.5M12 12v9M12 12L4 7.5" />
              </svg>
            </div>
            <h1 className="text-lg font-semibold" style={{ color: 'var(--wc-text-primary)' }}>{brand}</h1>
            <p className="mt-1 text-sm" style={{ color: 'var(--wc-text-secondary)' }}>
              {description || '使用 BasaltPass 单点登录以继续。'}
            </p>
          </div>
          {error && (
            <div className="rounded-xl px-3 py-2 text-sm" role="alert"
                 style={{ background: 'var(--wc-bg-error-subtle)', color: 'var(--wc-text-error)' }}>
              {error}
            </div>
          )}
          <Button variant="primary" size="lg" fullWidth loading={busy} onClick={handleLogin}>
            使用 BasaltPass 登录
          </Button>
          <p className="text-center text-xs" style={{ color: 'var(--wc-text-secondary)' }}>
            受 BasaltPass 身份服务保护
          </p>
        </Stack>
      </Card>
    </div>
  )
}
